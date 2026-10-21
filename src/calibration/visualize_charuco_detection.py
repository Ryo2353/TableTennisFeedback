"""
ChArUco検出結果を可視化するツール

使い方:
    python3 visualize_charuco_detection.py <画像パス> <列数> <行数> <出力名>

例:
    python3 visualize_charuco_detection.py ./stereo_top/pair_001.jpg 5 3 top_001

検出できたマーカー(緑の枠+ID)とChArUcoコーナー(赤い点+ID)を描画した画像を
保存する。何個検出できたかも表示する。
"""

import sys
import cv2
import cv2.aruco as aruco

MAX_DIM = 1600


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    cols, rows = int(sys.argv[2]), int(sys.argv[3])
    out_name = sys.argv[4]

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # マスの実寸は描画だけなら結果に影響しないため仮の値でよい
    board = aruco.CharucoBoard((cols, rows), 0.078, 0.0585, dictionary)
    detector = aruco.CharucoDetector(board)

    img = cv2.imread(path)
    if img is None:
        print(f"[エラー] 画像を読み込めません: {path}")
        sys.exit(1)

    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    vis = img.copy()

    n_markers = 0 if marker_ids is None else len(marker_ids)
    n_corners = 0 if charuco_ids is None else len(charuco_ids)
    expected_corners = (cols - 1) * (rows - 1)

    if marker_ids is not None:
        aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if charuco_ids is not None:
        aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 0, 255))

    # 検出数のサマリーを画像上部に描画
    summary = f"markers: {n_markers} / corners: {n_corners} (expected corners: {expected_corners})"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (255, 255, 255), -1)
    cv2.putText(vis, summary, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    out_path = f"charuco_detected_{out_name}.png"
    cv2.imwrite(out_path, vis)

    print(f"保存しました: {out_path}")
    print(f"検出マーカー数: {n_markers}")
    print(f"検出コーナー数: {n_corners} (期待値: {expected_corners})")
    if charuco_ids is not None:
        print(f"検出コーナーID: {sorted(charuco_ids.ravel().tolist())}")
    if n_corners < expected_corners:
        print(f"[注意] 期待値より少ないコーナーしか検出できていません。撮影距離・角度・照明を見直してください。")


if __name__ == "__main__":
    main()