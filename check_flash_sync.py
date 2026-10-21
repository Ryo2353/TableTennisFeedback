# -*- coding: utf-8 -*-
"""
点滅テストの解析: 2台のカメラの「delta(録画開始から何枚目か)」が
本当に同じ瞬間を指しているかを確かめる。

考え方:
    ライトを点けた/消した瞬間は、2台で同時に起きる出来事。
    各カメラで「画面全体の明るさが急に変わったコマ」を探し、
    そのdelta(= bid - 録画開始時のbid)を2台で比べる。
      - 差が 0      → delta 0 同士は同じ瞬間(同期OK)
      - 差が ±1 以上 → 1コマ以上ずれている(数え始めがずれている)

    明るさが変わる途中のコマ(半分だけ明るい等)を使って、
    「何コマ目と何コマ目の間で変わったか」を小数で推定する(コマ未満のずれの目安)。
    ライト自体がふわっと点く場合もあるので、小数部分はあくまで目安。

使い方:
    python check_flash_sync.py <セッションフォルダ> [<セッションフォルダ> ...] [--fps 227]

    セッションフォルダ = capture_dual.py が作る YYYYMMDD_HHMMSS フォルダ
    (中に camera_a と camera_b がある)

例:
    python check_flash_sync.py "C:\\...\\data\\captures\\20261014_183012" "C:\\...\\data\\captures\\20261014_183245"
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np

BID_PATTERN = re.compile(r"^(\d+)_bid(\d+)_ts(\d+)", re.IGNORECASE)


def load_brightness(cam_dir):
    """各コマの (bid, 画面全体の平均の明るさ) を返す。bid順に並べる。"""
    rows = []
    for p in glob.glob(os.path.join(cam_dir, "*.bmp")):
        m = BID_PATTERN.match(os.path.basename(p))
        if not m:
            continue
        bid = int(m.group(2))
        # 1/4に縮小して読む(平均の明るさを見るだけなので十分で、速い)
        img = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if img is None:
            continue
        rows.append((bid, float(img.mean())))
    rows.sort()
    if not rows:
        return None, None
    bids = np.array([r[0] for r in rows])
    vals = np.array([r[1] for r in rows])
    return bids, vals


def find_events(deltas, vals):
    """
    明るさが急に変わった場所(点灯/消灯)を探す。
    返り値: [(向き '+'/'-', 変化した位置の推定delta(小数), 変化の大きさ), ...]
    """
    d = np.diff(vals)
    if len(d) == 0:
        return []
    med = np.median(d)
    noise = 1.4826 * np.median(np.abs(d - med)) + 1e-6
    thr = max(1.0, 8 * noise, 0.25 * np.max(np.abs(d)))

    big = np.where(np.abs(d) > thr)[0]
    # 連続するコマ(変化が2コマにまたがる場合)を1つの出来事にまとめる
    groups = []
    for i in big:
        if groups and i - groups[-1][-1] <= 2 and np.sign(d[i]) == np.sign(d[groups[-1][-1]]):
            groups[-1].append(i)
        else:
            groups.append([i])

    events = []
    for g in groups:
        i0, i1 = g[0], g[-1] + 1          # vals[i0] が変化前、vals[i1] が変化後
        before = vals[max(0, i0 - 3):i0 + 1].mean()
        after = vals[i1:i1 + 4].mean()
        mid = (before + after) / 2
        rising = after > before
        # 「変化後の側に入った最初のコマ」(明るさが中間値を越えたコマ)を探し、
        # その直前との境目(-0.5)を変化した位置とする。
        # 露光(1ms)はコマ間隔(約4.4ms)より短いので、途中の明るさのコマから
        # コマ未満の時刻を正確に出すことはできない。1コマ単位で判定し、
        # 複数回の点滅の平均でコマ未満のずれを見積もる。
        pos = deltas[i1] - 0.5
        for k in range(i0 + 1, i1 + 1):
            if (vals[k] > mid) if rising else (vals[k] < mid):
                pos = deltas[k] - 0.5
                break
        events.append(("+" if rising else "-", float(pos), float(after - before)))
    return events


def analyze_session(session_dir, fps):
    print("=" * 70)
    print(f"セッション: {session_dir}")
    results = {}
    for cam in ("camera_a", "camera_b"):
        bids, vals = load_brightness(os.path.join(session_dir, cam))
        if bids is None:
            print(f"  [{cam}] 画像が見つかりません")
            return None
        deltas = bids - bids[0]
        drops = int(np.sum(np.diff(bids) - 1))
        events = find_events(deltas, vals)
        results[cam] = events
        print(f"  [{cam}] コマ数 {len(bids)}  コマ落ち {drops}  明るさ {vals.min():.1f}〜{vals.max():.1f}  "
              f"点滅の検出 {len(events)}回")
        for sign, pos, size in events:
            kind = "点灯" if sign == "+" else "消灯"
            print(f"      {kind}  delta ≈ {pos:7.2f}  (明るさの変化 {size:+.1f})")

    ea, eb = results["camera_a"], results["camera_b"]
    if not ea or not eb:
        print("  → 点滅を検出できませんでした。ライトが写っているか、明るさの変化が十分か確認してください。")
        return None

    # 同じ向き(点灯/消灯)で、いちばん近いもの同士を組にする
    pairs = []
    used = set()
    for sa, pa, _ in ea:
        cands = [(abs(pb - pa), j, pb) for j, (sb, pb, _) in enumerate(eb) if sb == sa and j not in used]
        if not cands:
            continue
        dist, j, pb = min(cands)
        if dist <= 20:
            used.add(j)
            pairs.append((sa, pa, pb))

    if not pairs:
        print("  → 2台の点滅を対応付けられませんでした。")
        return None

    print("  2台の比較(B − A。プラスならBの方が後のdeltaで変化している):")
    diffs = []
    for sign, pa, pb in pairs:
        diff = pb - pa
        diffs.append(diff)
        kind = "点灯" if sign == "+" else "消灯"
        print(f"      {kind}: A {pa:7.2f}  B {pb:7.2f}  → 差 {diff:+.2f}コマ (約{diff / fps * 1000:+.1f}ms)")

    diffs = np.array(diffs)
    mean = float(diffs.mean())
    whole = int(np.round(mean))
    print(f"  まとめ: 差の平均 {mean:+.2f}コマ(約{mean / fps * 1000:+.1f}ms)  "
          f"出てきた差: {sorted(set(int(round(x)) for x in diffs))}")
    print("    ※1回ごとの差は1コマ単位でしか出ない。平均がコマ未満のずれの目安"
          "(点滅の回数が多いほど正確)")
    if whole == 0:
        print("  → 判定: 1コマ丸ごとのずれはなし。delta 0 同士はほぼ同じ瞬間(ずれは1コマ未満)。")
    else:
        print(f"  → 判定: 約{whole:+d}コマずれている。Bのdeltaに{-whole:+d}すると対応が合う。")
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", help="セッションフォルダ(camera_a, camera_b を含む)")
    parser.add_argument("--fps", type=float, default=227.0, help="ms換算に使うfps(目安)")
    args = parser.parse_args()

    summary = []
    for s in args.sessions:
        r = analyze_session(os.path.abspath(s), args.fps)
        summary.append((s, r))

    if len(summary) > 1:
        print("=" * 70)
        print("全セッションのまとめ(B − A の差の平均):")
        for s, r in summary:
            name = os.path.basename(os.path.normpath(s))
            print(f"  {name}: " + ("判定できず" if r is None else f"{r:+.2f}コマ"))
        vals = [r for _, r in summary if r is not None]
        if len(vals) > 1:
            wholes = sorted(set(int(np.round(v)) for v in vals))
            if wholes == [0]:
                print("  → どのセッションも1コマ丸ごとのずれはなし(差は測定の分解能=1コマ以内)。")
            elif len(wholes) == 1:
                print(f"  → どのセッションも同じく約{wholes[0]:+d}コマずれている。固定の補正で直せる。")
            else:
                print("  → セッションごとに1コマ単位のずれ方が違う。録画ごとにずれを確かめて直す仕組みが必要。")


if __name__ == "__main__":
    main()