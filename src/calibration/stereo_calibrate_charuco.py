"""
ステレオキャリブレーション(ChArUcoボード版)

ChArUcoボードは各コーナーにID(番号)が振られているため、チェッカーボードで
問題になっていた「対応点の順序が写真ごとにズレる」という問題は原理的に
発生しない。両カメラで検出できた共通IDのコーナーだけを対応点として使う。

前提:
    calibrate_single_charuco.py を、カメラ1・カメラ2それぞれのステレオペア用
    フォルダに対して実行済みで、各フォルダ(または --output で指定した先)に
    calibration_result.npz が保存されていること。

使い方:
    python3 stereo_calibrate_charuco.py <カメラ1フォルダ> <カメラ2フォルダ> \
        <列数> <行数> <1マスの実寸mm> <マーカーの実寸mm> \
        [--calib1 PATH] [--calib2 PATH] [--output PATH] \
        [--min-corners N] [除外するファイル名...]

例:
    python3 stereo_calibrate_charuco.py ./data/captures/20260918/camera_a ./data/captures/20260918/camera_b \
        7 5 53 40 --calib1 ./results/20260918/camera_a --calib2 ./results/20260918/camera_b \
        --output ./results/20260918/stereo_result.npz --min-corners 20

--calib1 / --calib2 : calibration_result.npz を探すフォルダ(省略時はカメラ1/2フォルダ自身)。
                       calibrate_single_charuco.py で出力先を分けた場合はここで指定する。
--output PATH        : stereo_result.npz の保存先(省略時は実行時のカレントフォルダに
                        stereo_result.npz として保存。従来通りの挙動)。
--min-corners N       : 両カメラで共通するコーナー数がN未満のペアを除外する(デフォルト4)。
                        再投影誤差が高い場合、コーナー数の少ない(検出が不安定な)ペアが
                        原因になっていないか確認するのに使う。
"""

import sys
import os
import numpy as np
import cv2
import cv2.aruco as aruco

MAX_DIM = 1600
VALID_EXT = {".jpg", ".jpeg", ".png"}
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def list_images(folder):
    """{ペア対応キー: フルパス} の辞書を返す。

    capture_dual.py が保存するファイル名は
    "snapshot_{tag}_bid{block_id}_ts{timestamp}.png" の形式で、
    block_id・timestamp はカメラごとに別々の値になるため、
    ファイル名全体では両カメラ間で一致しない。
    "_bid" より前の部分(snapshot_001 など)だけを対応キーとして使う。
    """
    result = {}
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in VALID_EXT:
            continue
        key = stem.split("_bid")[0] if "_bid" in stem else stem
        result[key.lower()] = os.path.join(folder, f)
    return result


def detect(path, detector):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 4:
        return None
    return corners.reshape(-1, 2), ids.ravel(), gray.shape[::-1]


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        sys.exit(1)

    folder1, folder2 = sys.argv[1], sys.argv[2]
    cols, rows = int(sys.argv[3]), int(sys.argv[4])
    square_mm = float(sys.argv[5])
    marker_mm = float(sys.argv[6])

    remaining = sys.argv[7:]

    def pop_option(flag, default):
        nonlocal remaining
        if flag in remaining:
            i = remaining.index(flag)
            value = remaining[i + 1]
            remaining = remaining[:i] + remaining[i + 2:]
            return value
        return default

    min_corners = int(pop_option("--min-corners", "4"))
    calib1_folder = pop_option("--calib1", folder1)
    calib2_folder = pop_option("--calib2", folder2)
    output_path = pop_option("--output", "stereo_result.npz")
    exclude = {name.lower() for name in remaining}

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((cols, rows), square_mm / 1000, marker_mm / 1000, dictionary)
    detector = aruco.CharucoDetector(board)
    board_obj_points = board.getChessboardCorners()  # id -> 3D座標(m)

    npz1_path = os.path.join(calib1_folder, "calibration_result.npz")
    npz2_path = os.path.join(calib2_folder, "calibration_result.npz")
    for p in (npz1_path, npz2_path):
        if not os.path.exists(p):
            print(f"[エラー] {p} が見つかりません。先に calibrate_single_charuco.py を実行してください。")
            sys.exit(1)

    d1, d2 = np.load(npz1_path), np.load(npz2_path)
    camera_matrix1, dist1 = d1["camera_matrix"], d1["dist_coeffs"]
    camera_matrix2, dist2 = d2["camera_matrix"], d2["dist_coeffs"]

    images1, images2 = list_images(folder1), list_images(folder2)
    common_stems = sorted(set(images1) & set(images2))

    objpoints, imgpoints1, imgpoints2 = [], [], []
    image_size = None
    used_stems = []

    for stem in common_stems:
        if stem in exclude:
            print(f"  [skip] 除外指定: {stem}")
            continue

        r1 = detect(images1[stem], detector)
        r2 = detect(images2[stem], detector)
        if r1 is None or r2 is None:
            print(f"  [skip] 片方または両方で検出失敗: {stem}")
            continue

        corners1, ids1, size1 = r1
        corners2, ids2, size2 = r2
        if image_size is None:
            image_size = size1

        # 両カメラで共通して検出できたIDだけを対応点として使う
        common_ids = np.intersect1d(ids1, ids2)
        if len(common_ids) < min_corners:
            print(f"  [skip] 共通コーナー数不足({len(common_ids)}個 < {min_corners}): {stem}")
            continue

        idx1 = {id_: i for i, id_ in enumerate(ids1)}
        idx2 = {id_: i for i, id_ in enumerate(ids2)}

        obj_pts = np.array([board_obj_points[i] for i in common_ids], dtype=np.float32)
        img_pts1 = np.array([corners1[idx1[i]] for i in common_ids], dtype=np.float32).reshape(-1, 1, 2)
        img_pts2 = np.array([corners2[idx2[i]] for i in common_ids], dtype=np.float32).reshape(-1, 1, 2)

        objpoints.append(obj_pts)
        imgpoints1.append(img_pts1)
        imgpoints2.append(img_pts2)
        used_stems.append(stem)
        print(f"  [OK]   ペア成立({len(common_ids)}コーナー共通): {stem}")

    print(f"\n使用するペア数: {len(used_stems)} / {len(common_stems)}")
    if len(used_stems) < 8:
        print("[警告] ペア数が少なめです(目安10組以上)。外部パラメータの精度が不安定になる可能性があります。")

    flags = cv2.CALIB_FIX_INTRINSIC
    ret, camera_matrix1, dist1, camera_matrix2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        camera_matrix1, dist1, camera_matrix2, dist2,
        image_size, criteria=CRITERIA, flags=flags,
    )

    print("\n=== ステレオキャリブレーション結果 ===")
    print("再投影誤差(RMS):", ret)
    print("回転行列 R:\n", R)
    # 内部でChArUcoボードをメートル単位(square_mm/1000)で作っているため、
    # Tはメートル単位で返ってくる。表示はmmに揃える。
    T_mm = T.ravel() * 1000
    print("並進ベクトル T (mm):\n", T_mm)
    print(f"2台のカメラ間の距離(Tのノルム): {np.linalg.norm(T_mm):.2f} mm")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        output_path,
        camera_matrix1=camera_matrix1, dist1=dist1,
        camera_matrix2=camera_matrix2, dist2=dist2,
        R=R, T=T, E=E, F=F, image_size=image_size, rms=ret,
    )
    print(f"\n結果を保存しました: {output_path} (Tはメートル単位で保存)")


if __name__ == "__main__":
    main()