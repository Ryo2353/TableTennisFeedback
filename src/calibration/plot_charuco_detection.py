"""
calibrate_single_charuco.py と同じ設定(列数・行数・マス目/マーカーの実寸)で
1枚の画像に対してChArUcoボードの検出を行い、検出結果を可視化する
(「①キャリブレーション」スライド用に、カメラA・カメラB両方の画像を
 同じ見た目で並べられるようにするためのスクリプト)。

使い方(calibrate_single_charuco.pyと同じ引数の並び):
    python plot_charuco_detection.py <画像ファイル> <列数> <行数> <1マスの実寸mm> <マーカーの実寸mm> [出力png]

例:
    python plot_charuco_detection.py data\\captures\\20260918_160840\\camera_b\\some_image.png 7 5 53 40 charuco_b.png

出力される画像は、元の画像に
    - 検出できたマーカーを緑の枠+青字のIDでオーバーレイ
    - 検出できたChArUcoコーナーを赤い点+赤字のIDでオーバーレイ
    - 上部に "markers: X / corners: Y (expected corners: Y)" というタイトル
を重ねたものになる(最初にキャリブレーションのスライドに入れた画像と同じ見た目)。
"""

import sys
import os
import cv2
import cv2.aruco as aruco

MAX_DIM = 1600


def main():
    if len(sys.argv) not in (6, 7):
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    cols, rows = int(sys.argv[2]), int(sys.argv[3])
    square_mm = float(sys.argv[4])
    marker_mm = float(sys.argv[5])
    out_path = sys.argv[6] if len(sys.argv) == 7 else \
        os.path.splitext(image_path)[0] + "_charuco_detected.png"

    if not os.path.exists(image_path):
        print(f"[エラー] 画像が見つかりません: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[エラー] 画像の読み込みに失敗しました: {image_path}")
        sys.exit(1)

    # calibrate_single_charuco.py と同じ縮小処理(検出条件を揃えるため)
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((cols, rows), square_mm / 1000, marker_mm / 1000, dictionary)
    detector = aruco.CharucoDetector(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    expected_corners = (cols - 1) * (rows - 1)
    n_markers = 0 if marker_ids is None else len(marker_ids)
    n_corners = 0 if charuco_ids is None else len(charuco_ids)
    print(f"検出結果: markers={n_markers}, corners={n_corners} (期待値: {expected_corners})")

    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if False else img.copy()

    # ---- マーカーを緑の枠+青字のIDで描画 ----
    if marker_ids is not None and len(marker_ids) > 0:
        aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
        # drawDetectedMarkersのID表示は小さいので、見やすいよう自前でも重ねて描く
        for corners, mid in zip(marker_corners, marker_ids.flatten()):
            pt = tuple(corners[0][0].astype(int))
            cv2.putText(vis, f"id={mid}", (pt[0] - 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2, cv2.LINE_AA)

    # ---- ChArUcoコーナーを赤い点+赤字のIDで描画 ----
    if charuco_ids is not None and len(charuco_ids) > 0:
        for pt, cid in zip(charuco_corners.reshape(-1, 2), charuco_ids.flatten()):
            x, y = int(pt[0]), int(pt[1])
            cv2.rectangle(vis, (x - 4, y - 4), (x + 4, y + 4), (0, 0, 255), 1)
            cv2.putText(vis, f"id={cid}", (x + 6, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # ---- 上部にタイトルバーを描画(元画像と同じ体裁) ----
    title = f"markers: {n_markers} / corners: {n_corners} (expected corners: {expected_corners})"
    bar_h = 40
    vis_with_title = cv2.copyMakeBorder(vis, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(vis_with_title, title, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, vis_with_title)
    print(f"保存しました: {out_path}")


if __name__ == "__main__":
    main()