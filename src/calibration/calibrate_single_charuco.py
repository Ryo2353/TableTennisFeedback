"""
単眼カメラキャリブレーション(ChArUcoボード版)

使い方:
    python3 calibrate_single_charuco.py <画像フォルダ> <列数> <行数> <1マスの実寸mm> <マーカーの実寸mm> [出力先フォルダ]

例:
    python3 calibrate_single_charuco.py ./mono_top 5 3 78 58.5
    python3 calibrate_single_charuco.py ./data/captures/20260918_160840/camera_a 7 5 53 40 ./results/20260918

フォルダ内の画像すべてに対してChArUcoボードを検出し、cv2.calibrateCamera を
実行して内部パラメータと再投影誤差を表示・保存する。

出力先フォルダを省略した場合は、従来通り <画像フォルダ>/calibration_result.npz に保存する
(撮影データとキャリブレーション結果を分けたい場合は、6番目の引数で出力先を指定する)。
"""

import sys
import os
import numpy as np
import cv2
import cv2.aruco as aruco

MAX_DIM = 1600
VALID_EXT = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    )


def main():
    if len(sys.argv) not in (6, 7):
        print(__doc__)
        sys.exit(1)

    folder = sys.argv[1]
    cols, rows = int(sys.argv[2]), int(sys.argv[3])
    square_mm = float(sys.argv[4])
    marker_mm = float(sys.argv[5])
    # 出力先フォルダ(省略時は画像フォルダと同じ場所、従来通りの挙動)
    out_folder = sys.argv[6] if len(sys.argv) == 7 else folder

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((cols, rows), square_mm / 1000, marker_mm / 1000, dictionary)
    detector = aruco.CharucoDetector(board)

    paths = list_images(folder)
    if not paths:
        print(f"[エラー] {folder} に画像が見つかりません")
        sys.exit(1)

    print(f"{len(paths)}枚の画像を確認します...")

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    used_files = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] 読み込み失敗: {os.path.basename(path)}")
            continue

        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < 4:
            n = 0 if charuco_ids is None else len(charuco_ids)
            print(f"  [skip] 検出コーナー数不足({n}個): {os.path.basename(path)}")
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        used_files.append(path)
        print(f"  [OK]   検出成功({len(charuco_ids)}コーナー): {os.path.basename(path)}")

    print(f"\n検出成功: {len(used_files)} / {len(paths)} 枚")

    if len(used_files) < 4:
        print("[エラー] キャリブレーションに必要な枚数に届いていません。中止します。")
        sys.exit(1)

    # --- ChArUcoの検出結果を、calibrateCamera用のobjpoints/imgpointsに変換 ---
    objpoints = []
    imgpoints = []
    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        objpoints.append(obj_pts)
        imgpoints.append(img_pts)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("\n=== キャリブレーション結果 ===")
    print("再投影誤差(RMS):", ret)
    print("カメラ行列(内部パラメータ):\n", camera_matrix)
    print("レンズ歪み係数:\n", dist_coeffs.ravel())

    print("\n=== 画像ごとの再投影誤差 ===")
    per_image_errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        detected = np.asarray(imgpoints[i]).reshape(-1, 2)
        reprojected = np.asarray(proj).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((detected - reprojected) ** 2, axis=1))))
        per_image_errors.append(error)
        print(f"  {os.path.basename(used_files[i])}: {error:.4f} px")

    print(f"\n平均再投影誤差: {np.mean(per_image_errors):.4f} px")
    print(f"最大再投影誤差: {np.max(per_image_errors):.4f} px "
          f"({os.path.basename(used_files[int(np.argmax(per_image_errors))])})")

    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "calibration_result.npz")
    np.savez(
        out_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms=ret,
    )
    print(f"\n結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()