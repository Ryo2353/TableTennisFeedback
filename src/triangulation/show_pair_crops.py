# -*- coding: utf-8 -*-
"""
diagnose_triangulation.py が出した diagnose_points.csv を使って、
「2台で組にしたコマ」の切り出し画像(crops)をカメラA・Bで横に並べた一覧画像を作る。
再投影誤差が大きい組で、2台が本当に同じボールを見ているかを目で確かめるため。

使い方:
    python show_pair_crops.py <録画フォルダ> [--worst 24] [--best 12]

例:
    python show_pair_crops.py data\\captures\\20261016_190114\\rec02

出力:
    <録画フォルダ>\\pair_crops_worst.png  誤差が大きい組(上位 --worst 個)
    <録画フォルダ>\\pair_crops_best.png   誤差が小さい組(上位 --best 個。比較用)

各マスの左がカメラA、右がカメラB。上の数字は delta と再投影誤差(A/B、画素)。
"""

import argparse
import csv
import glob
import json
import os
import re

import cv2
import numpy as np

CELL = 120   # 切り出し画像1枚の表示の大きさ(画素)
PAT = re.compile(r"^(\d+)_bid(\d+)_")


def crop_index(rec_dir, cam):
    """crops フォルダの画像を delta → パス の辞書にする"""
    with open(os.path.join(rec_dir, "session_info.json"), encoding="utf-8") as f:
        _info = json.load(f)
        first_bid = min(_info["camera_a"]["first_bid"], _info["camera_b"]["first_bid"])   # 2台共通の基準(bid で組にする)
    idx = {}
    for p in glob.glob(os.path.join(rec_dir, cam, "crops", "*.png")):
        m = PAT.match(os.path.basename(p))
        if m:
            idx[int(m.group(2)) - first_bid] = p
    return idx


def tile(path):
    img = cv2.imread(path) if path else None
    if img is None:
        img = np.zeros((CELL, CELL, 3), np.uint8)
        cv2.putText(img, "no crop", (10, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(img, (CELL // 2, CELL // 2), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
    return img


def sheet(rows, idx_a, idx_b, out_path, cols=4):
    cells = []
    for r in rows:
        d = int(r["delta"])
        pair = np.hstack([tile(idx_a.get(d)), np.full((CELL, 4, 3), 255, np.uint8), tile(idx_b.get(d))])
        head = np.zeros((26, pair.shape[1], 3), np.uint8)
        cv2.putText(head, f"d={d} err {float(r['err_a_px']):.0f}/{float(r['err_b_px']):.0f}px",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cell = np.vstack([head, pair])
        cells.append(cv2.copyMakeBorder(cell, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(60, 60, 60)))
    if not cells:
        return
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    cv2.imwrite(out_path, grid)
    print(f"保存しました: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("--worst", type=int, default=24)
    ap.add_argument("--best", type=int, default=12)
    args = ap.parse_args()

    with open(os.path.join(args.rec_dir, "diagnose_points.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_err"] = max(float(r["err_a_px"]), float(r["err_b_px"]))
    rows.sort(key=lambda r: r["_err"])

    idx_a = crop_index(args.rec_dir, "camera_a")
    idx_b = crop_index(args.rec_dir, "camera_b")
    sheet(rows[::-1][:args.worst], idx_a, idx_b, os.path.join(args.rec_dir, "pair_crops_worst.png"))
    sheet(rows[:args.best], idx_a, idx_b, os.path.join(args.rec_dir, "pair_crops_best.png"))


if __name__ == "__main__":
    main()