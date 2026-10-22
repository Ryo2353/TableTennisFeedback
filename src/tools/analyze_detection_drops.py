# -*- coding: utf-8 -*-
r"""
全コマ保存した画像を使って、「ボールが写っているはずなのに検出されなかったコマ」が
なぜ落とされたのかを調べる。

検出のしかたは capture_live_detect.py と同じ(3フレーム差分 → 二値化 → 膨張・収縮 → 輪郭 →
面積・円形度・明るさの条件)。縮小率を変えて(既定は 0.5 と 1.0)両方で調べる。

「写っているはずのコマ」の決め方:
    検出できたコマとコマの間が短い(既定 30コマ以内)とき、その間のコマにもボールは写っているはず、とみなす。
    前後の検出位置から、そのコマでのボールの位置を直線で予想し、そのまわり(既定 半径80画素)の輪郭を調べる。

落とされた理由(予想位置のまわりで一番大きい輪郭について、最初に引っかかった条件):
    差分なし        … 予想位置のまわりに、差分(動いた所)がまったく出ていない
    面積が小さい    … 差分は出たが小さすぎる(細い三日月など)
    面積が大きい    … 大きすぎる(手・体など、ほかの物とつながった)
    円形度が低い    … 丸くない(三日月・細長い)
    明るさが足りない … 重心のまわりが暗い(重心がボールの外にずれた)

使い方(プロジェクトのいちばん上のフォルダで):
    python src\tools\analyze_detection_drops.py data\captures\20260918\hitball6\camera_a

    オプション:
      --scales 0.5 1.0     調べる縮小率(既定 0.5 と 1.0)
      --max-gap 30         「写っているはず」とみなす途切れの長さの上限(コマ)
      --examples 12        保存する例の画像の数
      --min-area 30 --min-circ 0.1 など   検出の条件を変えて試す(どれだけ途切れが減るか見る)

出力(指定したカメラのフォルダの中に drop_analysis/ を作る):
    summary.txt                 画面に出したまとめと同じもの
    examples_scale0.5.png など  落とされたコマの例(左: 元の画像、右: 3フレーム差分の結果)
"""

import argparse
import glob
import os
import re
from collections import Counter

import cv2
import numpy as np

# ---- 検出条件(capture_live_detect.py と同じ) ----
MIN_AREA = 100
MAX_AREA = 4000
MIN_CIRCULARITY = 0.3
MIN_BRIGHTNESS = 150
DIFF_THRESH = 50
KERNEL = np.ones((3, 3), np.uint8)

REASONS = ["差分なし", "面積が小さい", "面積が大きい", "円形度が低い", "明るさが足りない"]
PAT_BID = re.compile(r"_bid(\d+)_")


def list_frames(folder):
    """画像を撮った順に並べる(ファイル名の bid を優先、なければ名前順)"""
    paths = [p for p in glob.glob(os.path.join(folder, "*"))
             if p.lower().endswith((".bmp", ".png")) and "snapshot" not in os.path.basename(p)]

    def key(p):
        m = PAT_BID.search(os.path.basename(p))
        return (int(m.group(1)) if m else 0, os.path.basename(p))
    return sorted(paths, key=key)


def contours_with_reason(prev, cur, nxt, scale):
    """3フレーム差分の輪郭すべてについて、(cx, cy, 面積, 円形度, 明るさ, 理由 or None) を返す。座標・面積は元の大きさ"""
    diff = cv2.bitwise_and(cv2.absdiff(prev, cur), cv2.absdiff(cur, nxt))
    _, binary = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, KERNEL, iterations=2)
    binary = cv2.erode(binary, KERNEL, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    s2 = scale * scale
    r = max(2, int(round(5 * scale)))
    h, w = cur.shape[:2]
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        per = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (per ** 2) if per > 0 else 0
        patch = cur[max(0, int(cy) - r):min(h, int(cy) + r), max(0, int(cx) - r):min(w, int(cx) + r)]
        bright = float(patch.mean()) if patch.size else 0.0
        if area < MIN_AREA * s2:
            reason = "面積が小さい"
        elif area > MAX_AREA * s2:
            reason = "面積が大きい"
        elif circ < MIN_CIRCULARITY:
            reason = "円形度が低い"
        elif bright < MIN_BRIGHTNESS:
            reason = "明るさが足りない"
        else:
            reason = None
        out.append((cx / scale, cy / scale, area / s2, circ, bright, reason))
    return out, binary


def run_scale(paths, scale):
    """全コマを検出して、コマごとの輪郭一覧と差分画像の取り出し方を返す"""
    imgs = {}

    def load(i):
        if i not in imgs:
            g = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
            imgs[i] = g if scale == 1.0 else cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            for k in [k for k in imgs if k < i - 2]:
                del imgs[k]
        return imgs[i]

    frames = {}
    for i in range(1, len(paths) - 1):
        cs, _ = contours_with_reason(load(i - 1), load(i), load(i + 1), scale)
        frames[i] = cs
    return frames


def best_detection(cs):
    ok = [c for c in cs if c[5] is None]
    return max(ok, key=lambda c: c[2]) if ok else None


def runs_of(idx):
    if not idx:
        return []
    runs, cur = [], 1
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def analyze(paths, scale, max_gap, radius):
    frames = run_scale(paths, scale)
    det = {i: best_detection(cs) for i, cs in frames.items()}
    det_idx = sorted(i for i, d in det.items() if d is not None)
    runs = runs_of(det_idx)

    # 途切れが短い所の、間のコマを調べる
    missed = []   # (コマ番号, 予想位置, 理由, 近くの輪郭)
    for a, b in zip(det_idx, det_idx[1:]):
        gap = b - a - 1
        if gap < 1 or gap > max_gap:
            continue
        pa, pb = np.array(det[a][:2]), np.array(det[b][:2])
        if np.linalg.norm(pb - pa) > 400:      # 前後で位置が大きく飛ぶなら、別の物の可能性があるので使わない
            continue
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            exp = pa + t * (pb - pa)
            near = [c for c in frames.get(i, []) if np.hypot(c[0] - exp[0], c[1] - exp[1]) <= radius]
            if not near:
                missed.append((i, exp, "差分なし", None))
            else:
                c = max(near, key=lambda c: c[2])
                missed.append((i, exp, c[5] or "条件は通過(別の物を選んだ)", c))
    return frames, det_idx, runs, missed


def example_sheet(paths, scale, missed, n, out_path):
    """落とされたコマの例: 左に元の画像、右に3フレーム差分の結果(どちらも予想位置のまわり)"""
    if not missed:
        return
    step = max(1, len(missed) // n)
    picks = missed[::step][:n]
    half = 60
    cells = []
    for i, exp, reason, c in picks:
        g = [cv2.imread(paths[k], cv2.IMREAD_GRAYSCALE) for k in (i - 1, i, i + 1)]
        sm = g if scale == 1.0 else [cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for x in g]
        _, binary = contours_with_reason(sm[0], sm[1], sm[2], scale)
        if scale != 1.0:
            binary = cv2.resize(binary, (g[1].shape[1], g[1].shape[0]), interpolation=cv2.INTER_NEAREST)
        x, y = int(exp[0]), int(exp[1])
        H, W = g[1].shape
        x0, y0 = max(0, x - half), max(0, y - half)
        x1, y1 = min(W, x + half), min(H, y + half)
        a = cv2.copyMakeBorder(g[1][y0:y1, x0:x1], 0, 2 * half - (y1 - y0), 0, 2 * half - (x1 - x0), cv2.BORDER_CONSTANT)
        b = cv2.copyMakeBorder(binary[y0:y1, x0:x1], 0, 2 * half - (y1 - y0), 0, 2 * half - (x1 - x0), cv2.BORDER_CONSTANT)
        pair = np.hstack([a, np.full((2 * half, 3), 255, np.uint8), b])
        pair = cv2.resize(pair, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        pair = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
        label = {"差分なし": "no diff", "面積が小さい": "area small", "面積が大きい": "area large",
                 "円形度が低い": "low circ", "明るさが足りない": "dark"}.get(reason, "other")
        info = f"#{i} {label}"
        if c is not None:
            info += f" a={c[2]:.0f} c={c[3]:.2f} b={c[4]:.0f}"
        head = np.zeros((26, pair.shape[1], 3), np.uint8)
        cv2.putText(head, info, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(np.vstack([head, pair]))
    cols = 3
    rows = []
    for k in range(0, len(cells), cols):
        row = cells[k:k + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack([np.hstack([c, np.zeros((c.shape[0], 8, 3), np.uint8)]) for c in row]))
    cv2.imwrite(out_path, np.vstack([np.vstack([r, np.zeros((8, r.shape[1], 3), np.uint8)]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="1台分の画像フォルダ(例: data\\captures\\20260918\\hitball6\\camera_a)")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--max-gap", type=int, default=30)
    ap.add_argument("--radius", type=float, default=80, help="予想位置のまわりを調べる半径(画素、元の大きさ)")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--min-area", type=float, default=MIN_AREA, help=f"面積の下限(元の大きさで。今は {MIN_AREA})")
    ap.add_argument("--min-circ", type=float, default=MIN_CIRCULARITY, help=f"円形度の下限(今は {MIN_CIRCULARITY})")
    ap.add_argument("--min-bright", type=float, default=MIN_BRIGHTNESS, help=f"明るさの下限(今は {MIN_BRIGHTNESS})")
    ap.add_argument("--diff-thresh", type=float, default=DIFF_THRESH, help=f"差分の二値化のしきい値(今は {DIFF_THRESH})")
    args = ap.parse_args()

    # 条件を変えて試せるようにする(capture_live_detect.py は変わらない。良い値が見つかったら向こうに反映する)
    globals().update(MIN_AREA=args.min_area, MIN_CIRCULARITY=args.min_circ,
                     MIN_BRIGHTNESS=args.min_bright, DIFF_THRESH=args.diff_thresh)

    paths = list_frames(args.folder)
    if len(paths) < 3:
        print("画像が3枚未満です。フォルダを確認してください。")
        return
    out_dir = os.path.join(args.folder, "drop_analysis")
    os.makedirs(out_dir, exist_ok=True)
    lines = [f"フォルダ: {args.folder}", f"画像: {len(paths)} 枚",
             f"条件: 面積 {args.min_area:g}以上 / 円形度 {args.min_circ:g}以上 / 明るさ {args.min_bright:g}以上 / 差分しきい値 {args.diff_thresh:g}"]

    for scale in args.scales:
        frames, det_idx, runs, missed = analyze(paths, scale, args.max_gap, args.radius)
        lines.append(f"\n=== 縮小率 {scale} ===")
        lines.append(f"検出できたコマ: {len(det_idx)} / {len(frames)}")
        if runs:
            lines.append(f"続けて検出できた長さ: 最長 {max(runs)} コマ、平均 {sum(runs) / len(runs):.1f} コマ、{len(runs)} 回に分かれている")
        lines.append(f"途切れ({args.max_gap}コマ以内)の間で、写っているはずなのに落とされたコマ: {len(missed)}")
        cnt = Counter(m[2] for m in missed)
        for r in REASONS + ["条件は通過(別の物を選んだ)"]:
            if cnt[r]:
                lines.append(f"  {r:<14} {cnt[r]:>4} コマ ({100 * cnt[r] / len(missed):.0f}%)")
        near = [m[3] for m in missed if m[3] is not None]
        if near:
            a = np.array([c[2] for c in near])
            ci = np.array([c[3] for c in near])
            lines.append(f"  落とされた輪郭の面積: 中央値 {np.median(a):.0f}(条件 {args.min_area:g}〜{MAX_AREA})")
            lines.append(f"  落とされた輪郭の円形度: 中央値 {np.median(ci):.2f}(条件 {args.min_circ:g} 以上)")
        ex = os.path.join(out_dir, f"examples_scale{scale}.png")
        example_sheet(paths, scale, missed, args.examples, ex)
        if missed:
            lines.append(f"  例の画像: {ex}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()