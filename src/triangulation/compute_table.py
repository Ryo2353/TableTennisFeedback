"""
solvePnPで「カメラA基準の座標系」→「卓球台基準の座標系」への変換を求める。

三角測量(triangulate.py)が返す3D座標はカメラA基準(カメラAの光学中心が原点)。
これを卓球台基準(マシン側エンドライン左角が原点)の座標に変換するために、
卓球台上の実座標が分かっている点(4隅など)とカメラA画像上のピクセル座標の対応から、
カメラAの卓球台に対する位置・姿勢(R, t)をsolvePnPで求める。

前提:
    pick_points.py で、卓球台の4隅を以下の順序でカメラAの画像からクリックし、
    1つの *_points.json(4点)として保存しておくこと。
        1. エンドライン左(原点)
        2. エンドライン右
        3. プリント切れ目・左(本物の卓球台では、ネット際ラインとサイドラインの交点などに読み替える)
        4. プリント切れ目・右
    4隅の実座標(mm)は、このファイル内の TABLE_CORNERS_MM を実際の計測値に合わせて書き換えること。

使い方:
    python compute_table_pose.py <4隅points.json> <カメラAのcalibration_result.npz> <出力先.npz>

例:
    python compute_table_pose.py results/20260918/precision_check/corners/table_corners_points.json \
        results/20260918/camera_a/calibration_result.npz results/20260918/table_transform.npz

出力(table_transform.npz)には R, t が保存され、triangulate_points.py の --table-transform で使う。
"""

import sys
import json
import numpy as np
import cv2

# 卓球台(プリント)の実座標(mm)。原点=マシン側エンドライン左角。
# X = 116.5cm(プリント横幅)、Y = 84cm(原点からプリント切れ目までの距離)、
# プリント左端は原点に一致、という実測値に基づく。
TABLE_CORNERS_MM = np.array([
    [0.0, 0.0, 0.0],        # 1. エンドライン左(原点)
    [1165.0, 0.0, 0.0],     # 2. エンドライン右
    [0.0, 840.0, 0.0],      # 3. プリント切れ目・左
    [1165.0, 840.0, 0.0],   # 4. プリント切れ目・右
], dtype=np.float64)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    points_json_path, calib_npz_path, out_path = sys.argv[1:4]

    with open(points_json_path, "r", encoding="utf-8") as f:
        pts = json.load(f)
    if len(pts) != len(TABLE_CORNERS_MM):
        print(f"[エラー] {len(TABLE_CORNERS_MM)}点である必要がありますが、{len(pts)}点でした。"
              f" pick_points.pyでのクリック順序も、このファイルのTABLE_CORNERS_MMの順序と揃っているか確認してください。")
        sys.exit(1)
    img_points = np.array(pts, dtype=np.float64)

    d = np.load(calib_npz_path)
    K, dist = d["camera_matrix"], d["dist_coeffs"]

    ok, rvec, tvec = cv2.solvePnP(TABLE_CORNERS_MM, img_points, K, dist)
    if not ok:
        print("[エラー] solvePnPが解を見つけられませんでした。点の対応・順序を確認してください。")
        sys.exit(1)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    # 再投影誤差を確認(4隅をこのR,tで再投影し、実際のクリック位置とどれだけズレるか)
    proj, _ = cv2.projectPoints(TABLE_CORNERS_MM, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    errors = np.linalg.norm(proj - img_points, axis=1)

    print("=== solvePnP 結果(卓球台 -> カメラA) ===")
    print("回転行列 R:\n", R)
    print("並進ベクトル t (mm):\n", t)

    print("\n=== 再投影誤差(4隅、この変換が点の対応をどれだけ正確に説明できているか) ===")
    labels = ["エンドライン左", "エンドライン右", "プリント切れ目・左", "プリント切れ目・右"]
    for label, e in zip(labels, errors):
        print(f"  {label}: {e:.2f} px")
    print(f"RMS: {np.sqrt(np.mean(errors ** 2)):.2f} px")
    if np.sqrt(np.mean(errors ** 2)) > 5.0:
        print("[警告] 再投影誤差が大きめです。クリック位置や4隅の実座標にズレがないか確認してください。")

    np.savez(out_path, R=R, t=t)
    print(f"\n保存しました: {out_path}")


if __name__ == "__main__":
    main()