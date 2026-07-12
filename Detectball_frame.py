import cv2
import numpy as np
import glob
import os
import re
import time

# ============================================================
# 設定
# ============================================================
IMAGE_FOLDER = r"frames_smash_1"       # 映像ごとに書き換える
OUTPUT_FOLDER = r"output_detect_frame"

# ボール検出のフィルタ条件(Contourベース)
MIN_AREA = 100
MAX_AREA = 4000
MIN_CIRCULARITY = 0.3
MIN_BRIGHTNESS = 150

# ボールが映っている実際の範囲(映像ごとに書き換える)
BALL_START = 244
BALL_END = 261

# ============================================================
# 画像の読み込み(ファイルパス一覧のみ、実データはまだ読まない)
# ============================================================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")),
    key=lambda x: int(re.search(r'_(\d+)\.jpg$', x).group(1))
)
print(f"画像枚数: {len(image_paths)}")

if len(image_paths) < 3:
    print("画像が足りません。IMAGE_FOLDERのパスを確認してください。")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, "debug")
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ============================================================
# 検出ループ(三フレーム差分 + スライディングウィンドウ読み込み)
# ============================================================
ball_positions = []
area_list = []

# 処理時間計測用
time_imread = 0.0
time_diff = 0.0
time_binarize = 0.0
time_contour = 0.0

start_time = time.time()

# 最初の2枚だけ先に読み込んでおく(スライディングウィンドウの初期化)
frame_buffer = {}

t_read0 = time.time()
frame_buffer[0] = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
frame_buffer[1] = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)
t_read1 = time.time()
time_imread += (t_read1 - t_read0)

for i in range(1, len(image_paths) - 1):
    # ---- 画像読み込み(次の1枚だけ新規に読む) ----
    t_read0 = time.time()
    if (i + 1) not in frame_buffer:
        frame_buffer[i + 1] = cv2.imread(image_paths[i + 1], cv2.IMREAD_GRAYSCALE)
    t_read1 = time.time()
    time_imread += (t_read1 - t_read0)

    prev_frame = frame_buffer.get(i - 1)
    frame = frame_buffer.get(i)
    next_frame = frame_buffer.get(i + 1)

    if prev_frame is None or frame is None or next_frame is None:
        continue

    frame_num = int(re.search(r'_(\d+)\.jpg$', image_paths[i]).group(1))

    # ---- 三フレーム差分 ----
    t0 = time.time()
    diff1 = cv2.absdiff(prev_frame, frame)
    diff2 = cv2.absdiff(frame, next_frame)
    diff = cv2.bitwise_and(diff1, diff2)
    t1 = time.time()
    time_diff += (t1 - t0)

    # ---- 二値化 + ノイズ除去 ----
    _, binary = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)
    t2 = time.time()
    time_binarize += (t2 - t1)

    # デバッグ用画像保存(速度計測のため無効化中。必要な時だけコメント解除)
    debug_path = os.path.join(DEBUG_FOLDER, f"diff_{frame_num:04d}.jpg")
    cv2.imwrite(debug_path, binary)

    # ---- 輪郭検出 ----
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    t3 = time.time()
    time_contour += (t3 - t2)

    # ---- Contourベースでボール候補を判定(複数候補を記録) ----
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

        roi = frame[max(0, int(cy) - 5):min(frame.shape[0], int(cy) + 5),
                    max(0, int(cx) - 5):min(frame.shape[1], int(cx) + 5)]
        brightness = np.mean(roi)
        if brightness < MIN_BRIGHTNESS:
            continue

        candidates.append((cx, cy, area, circularity, brightness))

    # ---- 検出結果を記録(描画・保存は速度計測のため無効化中) ----
    if len(candidates) > 0:
        for cx, cy, area, circularity, brightness in candidates:
            bx, by = int(cx), int(cy)
            ball_positions.append((frame_num, bx, by, area, circularity))
            area_list.append(area)
        print(f"フレーム {frame_num:04d}: {len(candidates)}件検出 " +
              ", ".join([f"({int(c[0])},{int(c[1])},面積={c[2]:.1f})" for c in candidates]))
    else:
        print(f"フレーム {frame_num:04d}: 未検出")

    # ---- 結果画像の保存(確認したい時だけコメント解除) ----
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for cx, cy, area, circularity, brightness in candidates:
        bx, by = int(cx), int(cy)
        cv2.circle(vis, (bx, by), 15, (0, 0, 255), 2)
        cv2.putText(vis, f"({bx},{by})", (bx + 10, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, f"Frame: {frame_num:04d}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out_path = os.path.join(OUTPUT_FOLDER, f"result_{frame_num:04d}.jpg")
    cv2.imwrite(out_path, vis)

    # ---- 使い終わった古いフレームをバッファから削除(メモリ節約) ----
    if (i - 1) in frame_buffer:
        del frame_buffer[i - 1]

end_time = time.time()
elapsed = end_time - start_time

# ============================================================
# 結果サマリー
# ============================================================
print(f"\n--- 結果サマリー ---")
print(f"総フレーム数: {len(image_paths)}")

detected_frames = set(pos[0] for pos in ball_positions)
print(f"検出できたフレーム数（ユニーク）: {len(detected_frames)}")
print(f"検出された候補の総数（複数候補含む）: {len(ball_positions)}")

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

print(f"\n--- 処理時間の内訳 ---")
print(f"画像読み込み(imread): {time_imread*1000:.1f}ms (全体の{time_imread/elapsed*100:.1f}%)")
print(f"差分計算(absdiff): {time_diff*1000:.1f}ms (全体の{time_diff/elapsed*100:.1f}%)")
print(f"二値化+ノイズ除去: {time_binarize*1000:.1f}ms (全体の{time_binarize/elapsed*100:.1f}%)")
print(f"輪郭検出(findContours): {time_contour*1000:.1f}ms (全体の{time_contour/elapsed*100:.1f}%)")