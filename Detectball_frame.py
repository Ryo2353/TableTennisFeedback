import cv2
import numpy as np
import glob
import os
import re

# ============================================================
# 設定
# ============================================================
IMAGE_FOLDER = r"frames_smash_5"  # 例: r"C:\Users\xxx\images"
OUTPUT_FOLDER = r"output_detect_frame"  # 検出結果画像の保存先

# ボール検出のフィルタ条件（Contourベース）
MIN_AREA = 100      # これ未満の面積はノイズとみなす
MAX_AREA = 3500     # 実測値(1608.5~2975.5)に余裕を持たせた上限
MIN_CIRCULARITY = 0.3  # 円形度の下限（1に近いほど真円）
MIN_BRIGHTNESS = 150   # ボールは白いはずなので、輝度が低いものは除外

# ============================================================
# 画像の読み込み
# ============================================================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")),
    key=lambda x: int(re.search(r'_(\d+)\.jpg$', x).group(1))
)
print(f"画像枚数: {len(image_paths)}")

if len(image_paths) == 0:
    print("画像が見つかりません。IMAGE_FOLDERのパスを確認してください。")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, "debug")
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ============================================================
# 検出ループ（三フレーム差分）
# ============================================================
ball_positions = []  # (フレーム番号, x, y) を記録

for i, path in enumerate(image_paths):
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print(f"読み込み失敗: {path}")
        continue

    # 三フレーム差分には前後のフレームが必要なので最初と最後はスキップ
    if i == 0 or i == len(image_paths) - 1:
        continue

    prev_frame = cv2.imread(image_paths[i - 1], cv2.IMREAD_GRAYSCALE)
    next_frame = cv2.imread(image_paths[i + 1], cv2.IMREAD_GRAYSCALE)

    if prev_frame is None or next_frame is None:
        continue

    # 三フレーム差分
    diff1 = cv2.absdiff(prev_frame, frame)       # 前フレームとの差分
    diff2 = cv2.absdiff(frame, next_frame)       # 次フレームとの差分
    diff  = cv2.bitwise_and(diff1, diff2)        # 両方で動いている部分だけ残す

    # 二値化
    _, binary = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # ノイズ除去（膨張処理でボールの欠けを補う）
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)

    # デバッグ用：三フレーム差分の結果を保存
    debug_path = os.path.join(DEBUG_FOLDER, f"diff_{i:04d}.jpg")
    cv2.imwrite(debug_path, binary)

    # Contourベースでボール検出
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        roi = frame[max(0, int(cy)-5):min(frame.shape[0], int(cy)+5),
                    max(0, int(cx)-5):min(frame.shape[1], int(cx)+5)]
        brightness = np.mean(roi)
        if brightness < MIN_BRIGHTNESS:
            continue

        # 複数候補がある場合、最も面積が大きいものを採用
        if area > best_area:
            best_area = area
            best = (cx, cy, area, circularity, brightness)

    # 検出結果を描画
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if best is not None:
        bx, by = int(best[0]), int(best[1])
        ball_positions.append((i, bx, by))
        cv2.circle(vis, (bx, by), 15, (0, 0, 255), 2)
        cv2.putText(vis, f"Ball ({bx},{by})", (bx+10, by-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f"フレーム {i:04d}: ボール検出 ({bx}, {by}), 面積={best[2]:.1f}")
    else:
        print(f"フレーム {i:04d}: 未検出")

    # フレーム番号を表示
    cv2.putText(vis, f"Frame: {i:04d}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 結果を保存
    out_path = os.path.join(OUTPUT_FOLDER, f"result_{i:04d}.jpg")
    cv2.imwrite(out_path, vis)

# ============================================================
# 結果サマリー
# ============================================================
print(f"\n--- 結果サマリー ---")
print(f"総フレーム数: {len(image_paths)}")
print(f"検出フレーム数（全体）: {len(ball_positions)}")

ball_range = [pos for pos in ball_positions if 244 <= pos[0] <= 261]
false_positives = [pos for pos in ball_positions if pos[0] < 244 or pos[0] > 261]
ball_total = 261 - 244 + 1

print(f"ボールあり範囲(244-261)での検出数: {len(ball_range)} / {ball_total}フレーム")
print(f"ボールなし範囲での誤検出数: {len(false_positives)}")
print(f"正検出率: {len(ball_range)/ball_total*100:.1f}%")