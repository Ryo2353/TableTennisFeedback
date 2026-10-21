"""
calibration_result.npz (calibrate_single_charuco.py の出力) の camera_matrix/dist_coeffs
を使って、実際に1枚の画像を歪み補正し、補正前後を並べて確認するスクリプト。

やっていること:
    1. calibration_result.npz から camera_matrix, dist_coeffs を読み込む
    2. cv2.getOptimalNewCameraMatrix() で、補正後の画像がなるべく黒縁(無効領域)を
       含まないような新しいカメラ行列を計算する
    3. cv2.undistort() で実際に歪み補正をかける
    4. 補正前 / 補正後を左右に並べた画像を保存する
       (グリッド線を重ねると、直線が真っ直ぐになっているか見やすい)

使い方:
    python undistort_preview.py <calibration_result.npzのパス> <確認したい画像のパス> [出力先png]

例:
    python undistort_preview.py results\\20260918\\calibration_result_a.npz data\\captures\\hitball1\\camera_a\\000000_bid975_ts3765738517950.bmp results\\20260918\\undistort_preview_a.png
"""

import sys
import cv2
import numpy as np


def draw_grid(img, step=50, color=(0, 255, 0)):
    """直線が真っ直ぐになっているか目視確認しやすいように、グリッド線を重ねる。"""
    out = img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    npz_path = sys.argv[1]
    image_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "undistort_preview.png"

    data = np.load(npz_path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    print(f"camera_matrix:\n{camera_matrix}")
    print(f"dist_coeffs:\n{dist_coeffs}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[エラー] 画像が読み込めません: {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]

    # alpha=1 にすると、歪み補正で失われる周辺画素もできるだけ黒縁として残して
    # 全体を見せてくれる(0にすると黒縁を切り詰めて有効領域だけになる)。
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1
    )

    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    before = draw_grid(img)
    after = draw_grid(undistorted)

    cv2.putText(before, "Before (distorted)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(after, "After (undistorted)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    combined = np.hstack([before, after])
    cv2.imwrite(out_path, combined)
    print(f"\n保存しました: {out_path}")
    print("グリッド線が、Before側では中心から離れるほど歪んで見え、"
          "After側では真っ直ぐになっていれば補正が効いています。")


if __name__ == "__main__":
    main()