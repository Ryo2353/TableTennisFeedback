# -*- coding: utf-8 -*-
r"""
撮りながら検出(capture_live_detect.py)の結果から、ボールを3D(卓球台の座標、mm)で追跡し、
1球ごとの軌道に分ける。2球以上が同時に飛んでいても、別々の軌道として分ける。

triangulate_trajectory.py との違い:
    triangulate_trajectory.py は各コマで「2台の組を1つだけ」選び、時間で区切って軌道にしていた。
    そのため2球が同時に写ると、2球が1本に混ざっていた(10/23 rec02 の 5・6球目)。
    このスクリプトは、各コマで2台のつじつまが合う組を「全部」残し、軌道ごとに次の位置を予想して、
    その近くの点をつないでいく。

やり方(コマを時間の順に1つずつ処理し、未来のコマは見ない。撮りながら追跡にもそのまま使える):
    1. 各コマで、カメラAの候補 × カメラBの候補 のすべての組を三角測量し、再投影誤差が小さい組を残す
       (--max-reproj、標準 5画素)。誤差の小さい組から順に採用し、同じ候補は1コマで1回しか使わない
       (片方のボールの像ともう片方のボールの像を組にした「ありえない点」を減らすため)。
       3Dで近すぎる点(--merge、標準 30mm)は同じボールとして1つにまとめる
       (三日月が2つに割れて、同じボールが2つの候補になることがあるため)。
    2. 続いている軌道ごとに、直前の2点から次のコマの位置を予想する(速さ一定 + 重力)。
       予想に近い点から順に、軌道につなぐ。つなげる距離の上限は
           点が1つだけの軌道: --gate-new(標準 80mm)+ --speed-new(標準 50mm)× 空いたコマ数
           点が2つ以上の軌道: --gate(標準 40mm)+ --gate-grow(標準 15mm)× 空いたコマ数
       速さは直前の最大4点に直線を当てはめて求める(1点のぶれに強くするため)。
       予想した高さがボールの半径(--ball-r、標準 20mm)より低くなるときは、「台で弾んで上向きに
       跳ね返った位置」と「弾まずにそのまま落ちていく位置」(台の外・エッジに当たった球)の両方を予想し、
       近い方とつなぐ。台の近く(直前の高さ 100mm 以下)では、
       弾んで速さや向きが変わる分、つなげる距離に --gate-bounce(標準 40mm)を足す。
       (弾んだところで1球が2本に切れるのを防ぐため。10/31 rec03 の7球目、rec04 の4球目)
    3. どの軌道にもつながらなかった点からは、新しい軌道を始める。
    4. --max-gap(標準 10)コマより長く点が来なかった軌道は終わりにする。
    5. 最後に、点が --min-points(標準 10)より少ない軌道は捨てる(誤検出・ありえない点)。

使い方(プロジェクトのいちばん上のフォルダで。triangulate.py と同じフォルダに置く):
    python src\triangulation\track_balls.py <録画フォルダ> <stereo_result.npz> <table_transform.npz> <出力csv>
例:
    python src\triangulation\track_balls.py data\captures\20261023\20261023_182206\rec02 ^
        results\20261023\stereo_result.npz results\20261023\table_transform.npz ^
        results\20261023\rec02_tracks.csv

出力:
    <出力csv>        track, delta, ts_a, ts_b, x_mm, y_mm, z_mm, reproj_px
                     (delta は2台共通の基準bidからのコマ数。track は軌道の番号。捨てた軌道は入らない)
    <出力csv>.png    確認用の図(真上から見た位置 / 高さの変化。軌道ごとに色分け、捨てた点は灰色の×)
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from table_config import get_table
from triangulate import load_calibration, build_projection_matrices, triangulate, apply_table_transform
from triangulate_trajectory import (load_detections_by_bid, common_base_bid, offset_scan,
                                    candidates_of, reproj_errors)

FPS = 227.0
G_MM_PER_FRAME2 = 9810.0 / FPS ** 2        # 重力加速度(mm / コマ^2)、約 0.19


# ---------------- 1. 各コマの3D点 ----------------
def frame_points(ca, cb, P1, P2, K1, dist1, K2, dist2, R, T_mm, td, max_reproj, merge_mm, min_z):
    """1コマ分: 2台の候補の組から、つじつまの合う3D点(台の座標)の一覧 [(X(3,), 誤差, ia, ib)] を返す"""
    if not ca or not cb:
        return []
    pa = np.array([ca[i] for i in range(len(ca)) for _ in range(len(cb))], np.float64)
    pb = np.array([cb[j] for _ in range(len(ca)) for j in range(len(cb))], np.float64)
    idx = [(i, j) for i in range(len(ca)) for j in range(len(cb))]
    Xc = triangulate(P1, P2, K1, dist1, K2, dist2, pa, pb)
    errs = reproj_errors(Xc, pa, pb, K1, dist1, K2, dist2, R, T_mm)
    Xt = apply_table_transform(Xc, td["R"], td["t"])

    order = np.argsort(errs)
    used_a, used_b, out = set(), set(), []
    for k in order:
        e = float(errs[k])
        if e > max_reproj:
            break
        i, j = idx[k]
        if i in used_a or j in used_b:
            continue
        X = np.asarray(Xt[k], np.float64)
        if min_z is not None and X[2] < min_z:
            continue
        if any(np.linalg.norm(X - q[0]) < merge_mm for q in out):
            # 同じボールが割れた候補。どちらの候補も使用済みにする
            used_a.add(i)
            used_b.add(j)
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((X, e, i, j))
    return out


# ---------------- 2〜4. 追跡 ----------------
class Track:
    _next_id = 0

    def __init__(self, d, X, e):
        self.id = Track._next_id
        Track._next_id += 1
        self.ds, self.Xs, self.errs = [d], [X], [e]

    def predict(self, d, ball_r):
        """予想位置の候補の一覧と、空いたコマ数を返す。
        速さは直前の最大4点(8コマ以内)に直線を当てはめて求める(1点だけ位置がぶれても予想が大きく狂わないように)。
        台の面(ボールの中心で ball_r の高さ)より下になる予想のときは、
        「弾んで上向きに跳ね返った」と「弾まずにそのまま落ちていく」(台の端の外・エッジに当たった球)の
        両方を候補にする。点はどちらか近い方とつなぐ。"""
        dt = d - self.ds[-1]
        if len(self.ds) < 2:
            return [self.Xs[-1]], dt
        ds = np.array(self.ds[-4:], float)
        Xs = np.array(self.Xs[-4:])
        keep = ds >= ds[-1] - 8
        ds, Xs = ds[keep], Xs[keep]
        if len(ds) >= 3:
            v = np.array([np.polyfit(ds, Xs[:, k], 1)[0] for k in range(3)])
        else:
            v = (self.Xs[-1] - self.Xs[-2]) / (self.ds[-1] - self.ds[-2])
        p = self.Xs[-1] + v * dt
        p = p + np.array([0.0, 0.0, -0.5 * G_MM_PER_FRAME2 * dt * dt])
        if p[2] < ball_r and v[2] < 0:
            q = p.copy()
            q[2] = 2 * ball_r - p[2]
            return [q, p], dt
        return [p], dt

    def gate(self, dt, a):
        if len(self.ds) < 2:
            return a.gate_new + a.speed_new * dt
        g = a.gate + a.gate_grow * dt
        if self.Xs[-1][2] <= 100.0:
            g += a.gate_bounce
        return g

    def add(self, d, X, e):
        self.ds.append(d)
        self.Xs.append(X)
        self.errs.append(e)


def track(frames, a):
    """frames: [(d, [(X, err, ...), ...])] を時間の順に。返り値: (残した軌道, 捨てた軌道)"""
    active, finished = [], []
    for d, pts in frames:
        # 長く途切れた軌道を終わりにする
        still = []
        for t in active:
            (finished if d - t.ds[-1] > a.max_gap else still).append(t)
        active = still

        # 予想に近い順に、軌道と点をつなぐ(1対1)
        pairs = []
        for ti, t in enumerate(active):
            ps, dt = t.predict(d, a.ball_r)
            g = t.gate(dt, a)
            for pi, (X, e, *_ ) in enumerate(pts):
                dist = min(float(np.linalg.norm(X - p)) for p in ps)
                if dist <= g:
                    pairs.append((dist, ti, pi))
        pairs.sort()
        used_t, used_p = set(), set()
        for dist, ti, pi in pairs:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            active[ti].add(d, pts[pi][0], pts[pi][1])

        # つながらなかった点からは新しい軌道を始める
        for pi, (X, e, *_ ) in enumerate(pts):
            if pi not in used_p:
                active.append(Track(d, X, e))
    finished += active
    keep = [t for t in finished if len(t.ds) >= a.min_points]
    drop = [t for t in finished if len(t.ds) < a.min_points]
    keep.sort(key=lambda t: t.ds[0])
    return keep, drop


# ---------------- 図 ----------------
def plot(keep, drop, out_png, base_offset, table_w, table_l):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic")
                                   if f in have] + ["sans-serif"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    cmap = plt.get_cmap("tab10")
    for t in drop:
        X = np.array(t.Xs)
        ts = (np.array(t.ds) + base_offset) / FPS
        ax1.scatter(X[:, 0], X[:, 1], marker="x", s=12, c="#9CA3AF")
        ax2.scatter(ts, X[:, 2], marker="x", s=12, c="#9CA3AF")
    for k, t in enumerate(keep):
        X = np.array(t.Xs)
        ts = (np.array(t.ds) + base_offset) / FPS
        c = cmap(k % 10)
        ax1.plot(X[:, 0], X[:, 1], "-o", ms=3, lw=1, c=c, label=f"軌道{k + 1}({len(t.ds)}点)")
        ax2.plot(ts, X[:, 2], "-o", ms=3, lw=1, c=c)
        ax2.annotate(str(k + 1), (ts[0], X[0, 2]), fontsize=10, color=c)
    ax1.add_patch(plt.Rectangle((0, 0), table_w, table_l, fill=False, lw=2, ec="#333333"))
    ax1.set_xlabel("X (mm) 幅方向")
    ax1.set_ylabel("Y (mm) 長さ方向")
    ax1.set_title("真上から見た位置(枠 = 計測範囲、灰色の× = 捨てた点)")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="best")
    ax2.axhline(0, c="#333333", lw=1)
    ax2.set_xlabel("録画開始からの時間 (秒)")
    ax2.set_ylabel("Z (mm) 台の面からの高さ")
    ax2.set_title("高さの変化(数字 = 軌道の番号)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)


def main():
    ap = argparse.ArgumentParser(description="ボールを3Dで追跡し、1球ごとの軌道に分ける")
    ap.add_argument("rec_dir")
    ap.add_argument("stereo_npz")
    ap.add_argument("table_npz")
    ap.add_argument("out_csv")
    ap.add_argument("--max-reproj", type=float, default=5.0, help="この画素数より再投影誤差が大きい組は使わない")
    ap.add_argument("--merge", type=float, default=30.0, help="1コマの中でこれ(mm)より近い点は同じボールとしてまとめる")
    ap.add_argument("--min-z", type=float, default=-50.0, help="台の座標で Z がこれより低い点は使わない")
    ap.add_argument("--gate", type=float, default=40.0)
    ap.add_argument("--gate-grow", type=float, default=15.0)
    ap.add_argument("--gate-new", type=float, default=80.0)
    ap.add_argument("--speed-new", type=float, default=50.0)
    ap.add_argument("--ball-r", type=float, default=20.0, help="ボールの半径(mm)。弾む高さ(中心)")
    ap.add_argument("--gate-bounce", type=float, default=40.0, help="台の近くでつなげる距離に足す余裕(mm)")
    ap.add_argument("--max-gap", type=int, default=10, help="これより長く(コマ)点が来なかった軌道は終わりにする")
    ap.add_argument("--min-points", type=int, default=10, help="点がこれより少ない軌道は捨てる")
    ap.add_argument("--no-auto-shift", action="store_true", help="2台のコマのずれを自動で直さない")
    ap.add_argument("--table", default=None, help="台の名前(src/table_config.py)。図の枠だけに使う")
    a = ap.parse_args()

    det_a_path = os.path.join(a.rec_dir, "camera_a", "detections.json")
    det_b_path = os.path.join(a.rec_dir, "camera_b", "detections.json")
    info_path = os.path.join(a.rec_dir, "session_info.json")
    info = None
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    base = common_base_bid(info, [det_a_path, det_b_path])
    det_a = load_detections_by_bid(det_a_path, base)
    det_b = load_detections_by_bid(det_b_path, base)

    K1, dist1, K2, dist2, R, T = load_calibration(a.stereo_npz)
    P1, P2 = build_projection_matrices(K1, K2, R, T)
    T_arr = np.asarray(T, np.float64).reshape(3)
    T_mm = T_arr * 1000.0 if np.linalg.norm(T_arr) < 10 else T_arr
    print(f"[情報] キャリブレーション: {a.stereo_npz}(2台の距離 {np.linalg.norm(T_mm):.0f}mm)← 撮影した日のものか確認")
    td = np.load(a.table_npz)

    scan = offset_scan(det_a, det_b, P1, P2, K1, dist1, K2, dist2, R, T_mm)
    if scan:
        best_k = min(scan, key=lambda k: scan[k][0])
        if best_k != 0 and 0 in scan and scan[best_k][0] < 0.7 * scan[0][0]:
            print(f"[警告] 2台のコマの対応が {best_k:+d} コマずれている可能性(中央値 {scan[0][0]:.1f} → {scan[best_k][0]:.1f}画素)")
            if not a.no_auto_shift:
                print(f"       カメラBを {best_k:+d} コマずらして組み直します")
                det_b = {d + best_k: v for d, v in det_b.items()}
        else:
            print(f"[確認] 2台のコマのずれなし(k=0 の中央値 {scan.get(0, (float('nan'),))[0]:.1f}画素)")

    common = sorted(set(det_a) & set(det_b))
    frames, n_pts, n_multi = [], 0, 0
    for d in common:
        pts = frame_points(candidates_of(det_a[d]), candidates_of(det_b[d]), P1, P2, K1, dist1, K2, dist2,
                           R, T_mm, td, a.max_reproj, a.merge, a.min_z)
        n_pts += len(pts)
        n_multi += len(pts) >= 2
        frames.append((d, pts))
    print(f"[情報] 2台そろったコマ {len(common)}、つじつまの合う3D点 {n_pts}(2点以上あったコマ {n_multi})")

    keep, drop = track(frames, a)
    n_drop = sum(len(t.ds) for t in drop)
    print(f"[結果] 軌道 {len(keep)} 本(捨てた短い軌道 {len(drop)} 本・{n_drop} 点)")
    print("  番号  始め(秒)  終わり(秒)  点数  一番低いZ(mm)  X範囲(mm)        Y範囲(mm)")
    for k, t in enumerate(keep):
        X = np.array(t.Xs)
        print(f"  {k + 1:>4}  {t.ds[0] / FPS:8.2f}  {t.ds[-1] / FPS:9.2f}  {len(t.ds):4d}  {X[:, 2].min():12.0f}"
              f"  {X[:, 0].min():6.0f}〜{X[:, 0].max():<6.0f}  {X[:, 1].min():6.0f}〜{X[:, 1].max():<6.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    with open(a.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "delta", "ts_a", "ts_b", "x_mm", "y_mm", "z_mm", "reproj_px"])
        for k, t in enumerate(keep):
            for d, X, e in zip(t.ds, t.Xs, t.errs):
                w.writerow([k + 1, d, det_a[d].get("ts"), det_b[d].get("ts"),
                            f"{X[0]:.2f}", f"{X[1]:.2f}", f"{X[2]:.2f}", f"{e:.2f}"])
    png = os.path.splitext(a.out_csv)[0] + ".png"
    table_w, table_l, _ = get_table(a.table)
    plot(keep, drop, png, 0, table_w, table_l)
    print(f"保存: {a.out_csv}\n保存: {png}")


if __name__ == "__main__":
    main()