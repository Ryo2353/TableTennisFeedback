# -*- coding: utf-8 -*-
r"""
撮りながら検出(capture_live_detect.py)の切り出し画像を、目で確かめるための一覧を作る。

3種類の一覧を、録画フォルダ(rec01 など)の中に保存する:
    crops_only_a.png   カメラAだけで検出したコマ(Bでは検出なし)
    crops_only_b.png   カメラBだけで検出したコマ
    crops_both.png     2台とも検出したコマ(見本として、全体から均等に選ぶ)

各マス:
    赤い十字   = 記録した検出位置(元の解像度で中心を求め直した後)
    オレンジの丸 = 求め直す前の位置(差分の重心)。十字と離れていれば、求め直しでどれだけ動いたかが分かる
    上の文字   = bid、画面上の位置(u, v)、面積、そのコマの候補の数

使い方(プロジェクトのいちばん上のフォルダで):
    python src\tools\show_detection_crops.py <録画フォルダ> [--n 24]
例:
    python src\tools\show_detection_crops.py data\captures\20261023\20261023_182206\rec01
"""

import argparse
import glob
import json
import os
import re

import cv2
import numpy as np

CELL = 300
PAT = re.compile(r"_bid(\d+)_")


def load(rec, cam):
    with open(os.path.join(rec, cam, "detections.json"), encoding="utf-8") as f:
        dets = {d["bid"]: d for d in json.load(f) if d.get("bid") is not None}
    crops = {}
    for p in glob.glob(os.path.join(rec, cam, "crops", "*.png")):
        m = PAT.search(os.path.basename(p))
        if m:
            crops[int(m.group(1))] = p
    return dets, crops


def cell(det, crop_path, half):
    img = cv2.imread(crop_path) if crop_path else None
    if img is None:
        out = np.full((CELL, CELL, 3), 40, np.uint8)
        cv2.putText(out, "no crop", (10, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return out
    h, w = img.shape[:2]
    u, v = det["u"], det["v"]
    # 切り出しは撮影時に画面の端で切り詰められている(ずらしてはいない)ので、左上の位置は u-half, v-half を0で止めたもの
    x0, y0 = max(0, int(u) - half), max(0, int(v) - half)
    s = CELL / (2 * half)   # どの切り出しも同じ倍率で表示する(端で切り詰められたものも)
    big = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((CELL, CELL, 3), 40, np.uint8)
    canvas[:big.shape[0], :big.shape[1]] = big
    if "u_raw" in det:
        p = (int((det["u_raw"] - x0) * s), int((det["v_raw"] - y0) * s))
        cv2.circle(canvas, p, 6, (0, 165, 255), 2)
    c = (int((u - x0) * s), int((v - y0) * s))
    cv2.drawMarker(canvas, c, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    head = np.zeros((40, CELL, 3), np.uint8)
    cv2.putText(head, f"bid {det['bid']}  ({int(u)},{int(v)})", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(head, f"area {det.get('area', 0):.0f}  cand {det.get('num_candidates', '?')}",
                (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return np.vstack([head, canvas])


def sheet(items, out_path, cols=4):
    if not items:
        return False
    cells = [np.hstack([c, np.zeros((c.shape[0], 6, 3), np.uint8)]) for c in items]
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.vstack([np.hstack(row), np.zeros((6, cols * cells[0].shape[1], 3), np.uint8)]))
    cv2.imwrite(out_path, np.vstack(rows))
    return True


def pick(bids, n):
    bids = sorted(bids)
    if len(bids) <= n:
        return bids
    idx = np.linspace(0, len(bids) - 1, n).round().astype(int)
    return [bids[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("--n", type=int, default=24, help="1枚の一覧に並べる数")
    ap.add_argument("--crop-half", type=int, default=None,
                    help="撮影時の切り出しの半径。省略すると、いちばん大きい切り出し画像から求める")
    args = ap.parse_args()

    da, ca = load(args.rec_dir, "camera_a")
    db, cb = load(args.rec_dir, "camera_b")
    if args.crop_half is None:
        sizes = [max(cv2.imread(p).shape[:2]) for p in list(ca.values()) + list(cb.values())]
        args.crop_half = max(sizes) // 2 if sizes else 120
    print(f"切り出しの半径: {args.crop_half} 画素")
    only_a = set(da) - set(db)
    only_b = set(db) - set(da)
    both = set(da) & set(db)
    print(f"Aだけ: {len(only_a)}  Bだけ: {len(only_b)}  2台とも: {len(both)}")

    for name, bids, dets, crops in (("crops_only_a.png", only_a, da, ca), ("crops_only_b.png", only_b, db, cb)):
        out = os.path.join(args.rec_dir, name)
        if sheet([cell(dets[b], crops.get(b), args.crop_half) for b in pick(bids, args.n)], out):
            print(f"保存: {out}")
    # 2台とも:左にA、右にBを並べた組を一覧にする
    pairs = []
    for b in pick(both, args.n // 2):
        pairs.append(np.hstack([cell(da[b], ca.get(b), args.crop_half), np.full((CELL + 40, 4, 3), 255, np.uint8), cell(db[b], cb.get(b), args.crop_half)]))
    out = os.path.join(args.rec_dir, "crops_both.png")
    if sheet(pairs, out, cols=2):
        print(f"保存: {out}(各組 左がカメラA、右がカメラB)")

    # 中心の求め直しでどれだけ動いたか
    for cam, dets in (("A", da), ("B", db)):
        mv = [np.hypot(d["u"] - d["u_raw"], d["v"] - d["v_raw"]) for d in dets.values() if "u_raw" in d]
        if mv:
            mv = np.array(mv)
            print(f"カメラ{cam} 中心の求め直しで動いた距離(画素): 中央値 {np.median(mv):.1f}  90%点 {np.percentile(mv, 90):.1f}  最大 {mv.max():.1f}")


if __name__ == "__main__":
    main()