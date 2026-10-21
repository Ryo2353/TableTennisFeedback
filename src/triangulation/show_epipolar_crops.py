# -*- coding: utf-8 -*-
r"""
切り出し画像に「もう一方のカメラの視線が、このカメラの画面でどう見えるか」の線(エピポーラ線)を描く。

食い違いが小さいのに十字の位置が2台でそろっていない組について、
  「線がボールを縦に(または斜めに)貫いていて、十字がその線の上にあるから食い違いが0になる」
のかどうかを、目で確かめるためのスクリプト。

使い方(diagnose_triangulation.py と同じフォルダで。先に diagnose_triangulation.py を実行しておくこと):
    python show_epipolar_crops.py <録画フォルダ> <stereo_result.npz> [--worst 8] [--best 8]

例:
    python src\triangulation\show_epipolar_crops.py data\captures\20261016_190114\rec02 results\20261016\stereo_result.npz

出力(録画フォルダの中):
    epipolar_worst.png   食い違いが大きい組
    epipolar_best.png    食い違いが小さい組
    各マス: 左がカメラA、右がカメラB。
        赤い十字 = そのカメラで検出した位置
        水色の線 = もう一方のカメラの検出位置から伸ばした視線(このカメラの画面ではこう見える)
    十字が水色の線の上にあれば食い違いは小さく、線から離れていれば食い違いは大きい。
    画面にも、線の向き(水平から何度か)を表で出す。90度に近いほど縦向き。
"""

import argparse
import csv
import glob
import json
import os
import re

import cv2
import numpy as np

from triangulate import load_calibration

SCALE = 4           # 切り出し画像を何倍に拡大して描くか
PAT = re.compile(r"^(\d+)_bid(\d+)_")


def crop_index(rec_dir, cam):
    """crops フォルダの画像を delta → パス の辞書にする"""
    with open(os.path.join(rec_dir, "session_info.json"), encoding="utf-8") as f:
        first_bid = json.load(f)[cam]["first_bid"]
    idx = {}
    for p in glob.glob(os.path.join(rec_dir, cam, "crops", "*.png")):
        m = PAT.match(os.path.basename(p))
        if m:
            idx[int(m.group(2)) - first_bid] = p
    return idx


def ray_points(pt, K, dist):
    """画素の点 → そのカメラ座標系での視線上の点の列(奥行き 50mm〜20m)"""
    n = cv2.undistortPoints(np.array([[pt]], np.float64), K, dist).reshape(2)
    depths = np.geomspace(50, 20000, 4000)
    return np.stack([n[0] * depths, n[1] * depths, depths], axis=1)


def epiline_in_b(pt_a, K1, d1, K2, d2, rvec_b, T_mm):
    """カメラAの点から伸ばした視線を、カメラBの画面に写した点の列(レンズのゆがみ込み)"""
    X = ray_points(pt_a, K1, d1)
    p, _ = cv2.projectPoints(X, rvec_b, T_mm, K2, d2)
    return p.reshape(-1, 2)


def epiline_in_a(pt_b, K1, d1, K2, d2, R, T_mm):
    """カメラBの点から伸ばした視線を、カメラAの画面に写した点の列"""
    Xb = ray_points(pt_b, K2, d2)
    Xa = (R.T @ (Xb - T_mm.reshape(1, 3)).T).T
    p, _ = cv2.projectPoints(Xa, np.zeros(3), np.zeros(3), K1, d1)
    return p.reshape(-1, 2)


def line_info(curve, pt):
    """線と点の距離(画素)と、点のそばでの線の向き(水平から何度か、0〜90)"""
    d = np.linalg.norm(curve - np.asarray(pt), axis=1)
    i = int(np.argmin(d))
    j0, j1 = max(0, i - 3), min(len(curve) - 1, i + 3)
    v = curve[j1] - curve[j0]
    ang = abs(np.degrees(np.arctan2(v[1], v[0])))
    ang = 180 - ang if ang > 90 else ang
    return float(d[i]), float(ang)


def draw_cell(crop_path, pt, curve):
    """切り出し画像を拡大し、検出位置の十字と、もう一方のカメラの視線を描く"""
    img = cv2.imread(crop_path) if crop_path else None
    if img is None:
        img = np.full((80, 80, 3), 40, np.uint8)
        cv2.putText(img, "no crop", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    half = max(w, h) // 2
    x0 = max(0, int(pt[0]) - half)
    y0 = max(0, int(pt[1]) - half)
    big = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    loc = (curve - np.array([x0, y0])) * SCALE
    inside = (loc[:, 0] > -200) & (loc[:, 0] < w * SCALE + 200) & (loc[:, 1] > -200) & (loc[:, 1] < h * SCALE + 200)
    pts = loc[inside].round().astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(big, [pts.reshape(-1, 1, 2)], False, (255, 220, 0), 2, cv2.LINE_AA)
    c = (int((pt[0] - x0) * SCALE), int((pt[1] - y0) * SCALE))
    cv2.drawMarker(big, c, (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
    return big


def sheet(rows, idx_a, idx_b, calib, out_path, title):
    K1, d1, K2, d2, R, T_mm, rvec_b = calib
    cells = []
    print(f"\n=== {title} ===")
    print(f"{'delta':>7} {'食い違いA/B':>12} {'線の向きA':>9} {'線の向きB':>9}  (0度=横、90度=縦)")
    for r in rows:
        d = int(r["delta"])
        pa = (float(r["u_a"]), float(r["v_a"]))
        pb = (float(r["u_b"]), float(r["v_b"]))
        cb = epiline_in_b(pa, K1, d1, K2, d2, rvec_b, T_mm)
        ca = epiline_in_a(pb, K1, d1, K2, d2, R, T_mm)
        dist_a, ang_a = line_info(ca, pa)
        dist_b, ang_b = line_info(cb, pb)
        print(f"{d:>7} {dist_a:>5.0f}/{dist_b:<5.0f}px {ang_a:>7.0f}度 {ang_b:>7.0f}度")
        left = draw_cell(idx_a.get(d), pa, ca)
        right = draw_cell(idx_b.get(d), pb, cb)
        hh = max(left.shape[0], right.shape[0])
        left = cv2.copyMakeBorder(left, 0, hh - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        right = cv2.copyMakeBorder(right, 0, hh - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        pair = np.hstack([left, np.full((hh, 6, 3), 255, np.uint8), right])
        head = np.zeros((34, pair.shape[1], 3), np.uint8)
        cv2.putText(head, f"d={d}  err {dist_a:.0f}/{dist_b:.0f}px  line A {ang_a:.0f}deg  B {ang_b:.0f}deg",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(np.vstack([head, pair]))
    if not cells:
        return
    cw = max(c.shape[1] for c in cells)
    ch = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, cw - c.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for c in cells]
    cols = 2
    rows_img = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows_img.append(np.hstack([np.hstack([c, np.zeros((c.shape[0], 10, 3), np.uint8)]) for c in row]))
    rh = max(r.shape[0] for r in rows_img)
    out = np.vstack([np.vstack([r, np.zeros((10, r.shape[1], 3), np.uint8)]) for r in rows_img])
    cv2.imwrite(out_path, out)
    print(f"保存: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("stereo_npz")
    ap.add_argument("--worst", type=int, default=8)
    ap.add_argument("--best", type=int, default=8)
    args = ap.parse_args()

    csv_path = os.path.join(args.rec_dir, "diagnose_points.csv")
    if not os.path.exists(csv_path):
        print("diagnose_points.csv がありません。先に diagnose_triangulation.py を実行してください。")
        return
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_err"] = max(float(r["err_a_px"]), float(r["err_b_px"]))
    rows.sort(key=lambda r: r["_err"])

    K1, d1, K2, d2, R, T = load_calibration(args.stereo_npz)
    R = np.asarray(R, np.float64)
    T = np.asarray(T, np.float64).reshape(3)
    T_mm = T * 1000.0 if np.linalg.norm(T) < 10 else T     # diagnose_triangulation.py と同じ扱い
    rvec_b, _ = cv2.Rodrigues(R)
    calib = (np.asarray(K1, np.float64), np.asarray(d1, np.float64),
             np.asarray(K2, np.float64), np.asarray(d2, np.float64), R, T_mm, rvec_b)

    idx_a = crop_index(args.rec_dir, "camera_a")
    idx_b = crop_index(args.rec_dir, "camera_b")
    sheet(rows[::-1][:args.worst], idx_a, idx_b, calib,
          os.path.join(args.rec_dir, "epipolar_worst.png"), "食い違いが大きい組")
    sheet(rows[:args.best], idx_a, idx_b, calib,
          os.path.join(args.rec_dir, "epipolar_best.png"), "食い違いが小さい組")
    print("\n見方: 十字が水色の線の上にあれば、食い違いは小さく出る。")
    print("      線がボールを縦に貫いているなら、上端と下端のどちらを拾っても食い違いは0になる(奥行きだけ変わる)。")


if __name__ == "__main__":
    main()