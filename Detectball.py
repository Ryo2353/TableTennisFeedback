import cv2
import numpy as np
import glob
import os
import re
import time

# ============================================================
# 設定
# ============================================================
IMAGE_FOLDER = r"frames_smash_1"
OUTPUT_FOLDER = r"output_detect"

# MOG2の設定
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=50, detectShadows=False
)

# ボール検出のフィルタ条件（Contourベース、Detectball_frame.pyと統一）
MIN_AREA = 100
MAX_AREA = 4000
MIN_CIRCULARITY = 0.3
MIN_BRIGHTNESS = 150

# ============================================================
# 画像の読み込み
# ============================================================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")),
    key=lambda x: int(re.search(r'_(\d+)\.jpg$', x).group(1))
)
print(f"画像枚数: {len(image_paths)}")

if len(image_paths) == 0:
    print("画像が見つかりません。")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, "debug")
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ============================================================
# 検出ループ（MOG2 + Contourベース判定）
# ============================================================
ball_positions = []
area_list = []

start_time = time.time()

for i, path in enumerate(image_paths):
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        continue

    frame_num = int(re.search(r'_(\d+)\.jpg$', path).group(1))

    # MOG2で背景差分
    fg_mask = fgbg.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, ksize=5)

    # マスクを適用
    masked = np.zeros_like(frame)
    masked[fg_mask == 255] = frame[fg_mask == 255]

    # デバッグ用保存
    debug_path = os.path.join(DEBUG_FOLDER, f"diff_mog2_{frame_num:04d}.jpg")
    cv2.imwrite(debug_path, fg_mask)

    # Contourベースでボール候補を検出
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

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

        candidates.append((cx, cy, area, circularity, brightness))

    # 検出結果を描画
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if len(candidates) > 0:
        for cx, cy, area, circularity, brightness in candidates:
            bx, by = int(cx), int(cy)
            ball_positions.append((frame_num, bx, by, area, circularity))
            area_list.append(area)
            cv2.circle(vis, (bx, by), 15, (0, 0, 255), 2)
            cv2.putText(vis, f"({bx},{by})", (bx+10, by-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f"フレーム {frame_num:04d}: {len(candidates)}件検出 " +
              ", ".join([f"({int(c[0])},{int(c[1])},面積={c[2]:.1f})" for c in candidates]))
    else:
        print(f"フレーム {frame_num:04d}: 未検出")

    cv2.putText(vis, f"Frame: {frame_num:04d}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = os.path.join(OUTPUT_FOLDER, f"result_{frame_num:04d}.jpg")
    cv2.imwrite(out_path, vis)

end_time = time.time()
elapsed = end_time - start_time

# ============================================================
# 結果サマリー
# ============================================================
print(f"\n--- 結果サマリー（MOG2） ---")
print(f"総フレーム数: {len(image_paths)}")

detected_frames = set(pos[0] for pos in ball_positions)
print(f"検出できたフレーム数（ユニーク）: {len(detected_frames)}")
print(f"検出された候補の総数（複数候補含む）: {len(ball_positions)}")

BALL_START = 271  # 映像ごとに書き換える
BALL_END = 285
ball_total = BALL_END - BALL_START + 1

ball_range_frames = set(f for f in detected_frames if BALL_START <= f <= BALL_END)
false_positive_frames = set(f for f in detected_frames if f < BALL_START or f > BALL_END)

print(f"ボールあり範囲({BALL_START}-{BALL_END})での検出フレーム数: {len(ball_range_frames)} / {ball_total}フレーム")
print(f"ボールなし範囲での誤検出フレーム数: {len(false_positive_frames)}")
print(f"正検出率: {len(ball_range_frames)/ball_total*100:.1f}%")

if area_list:
    print(f"検出できた面積: 最小={min(area_list):.1f}, 最大={max(area_list):.1f}, 平均={sum(area_list)/len(area_list):.1f}")

print(f"\n総処理時間: {elapsed:.3f}秒")
print(f"1フレームあたり平均: {elapsed/len(image_paths)*1000:.2f}ms")