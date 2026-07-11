import cv2
import numpy as np
import glob
import os
import re

# ============================================================
# 設定
# ============================================================
IMAGE_FOLDER = r"frames_smash_0708"
OUTPUT_FOLDER = r"output_detect"

BALL_START = 265  # ボールが映り始めるフレーム番号
BALL_END   = 406  # ボールが映り終わるフレーム番号
BALL_TOTAL = BALL_END - BALL_START + 1

# MOG2の設定
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=50, detectShadows=False
)

# BlobDetectorの設定
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 255
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 5
params.maxArea = 2000
params.filterByCircularity = True
params.minCircularity = 0.3
params.filterByConvexity = True
params.minConvexity = 0.5
params.filterByInertia = True
params.minInertiaRatio = 0.08
detector = cv2.SimpleBlobDetector_create(params)

# ============================================================
# 画像の読み込み（正しいソート）
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
# 検出ループ（MOG2）
# ============================================================
ball_positions = []

for i, path in enumerate(image_paths):
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        continue

    # MOG2で背景差分
    fg_mask = fgbg.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, ksize=5)

    # マスクを適用
    masked = np.zeros_like(frame)
    masked[fg_mask == 255] = frame[fg_mask == 255]

    # masked画像の保存
    debug_path = os.path.join(DEBUG_FOLDER, f"masked_{i:04d}.jpg")
    cv2.imwrite(debug_path, masked)

    # diff画像（fg_mask）の保存
    diff_path = os.path.join(DEBUG_FOLDER, f"diff_mog2_{i:04d}.jpg")
    cv2.imwrite(diff_path, fg_mask)

    # BlobDetectorでボール検出
    keypoints = detector.detect(masked)

    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if len(keypoints) > 0:
        best = None
        best_val = -1
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            roi = frame[max(0, y-5):min(frame.shape[0], y+5),
                        max(0, x-5):min(frame.shape[1], x+5)]
            val = np.mean(roi)
            if val > best_val:
                best_val = val
                best = kp

        # 輝度フィルタ
        if best_val > 150:
            bx, by = int(best.pt[0]), int(best.pt[1])
            ball_positions.append((i, bx, by))
            cv2.circle(vis, (bx, by), 15, (0, 0, 255), 2)
            cv2.putText(vis, f"Ball ({bx},{by})", (bx+10, by-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            print(f"フレーム {i:04d}: ボール検出 ({bx}, {by})")
        else:
            print(f"フレーム {i:04d}: 未検出（輝度不足）")
    else:
        print(f"フレーム {i:04d}: 未検出")

    cv2.putText(vis, f"Frame: {i:04d}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = os.path.join(OUTPUT_FOLDER, f"result_{i:04d}.jpg")
    cv2.imwrite(out_path, vis)

# ============================================================
# 結果サマリー
# ============================================================
ball_range = [pos for pos in ball_positions if BALL_START <= pos[0] <= BALL_END]
false_positives = [pos for pos in ball_positions if pos[0] < BALL_START or pos[0] > BALL_END]
no_ball_total = len(image_paths) - BALL_TOTAL

print(f"\n--- 結果サマリー（MOG2） ---")
print(f"総フレーム数: {len(image_paths)}")
print(f"検出フレーム数（全体）: {len(ball_positions)}")
print(f"ボールあり範囲({BALL_START}-{BALL_END})での検出数: {len(ball_range)} / {BALL_TOTAL}フレーム")
print(f"ボールなし範囲での誤検出数: {len(false_positives)}")
print(f"正検出率: {len(ball_range)/BALL_TOTAL*100:.1f}%")
print(f"誤検出率: {len(false_positives)/no_ball_total*100:.1f}%")