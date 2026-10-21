import cv2
import numpy as np
import glob
import os
import re
import time
import json
import sys

# ============================================================
# 設定
# ============================================================
# コマンドライン引数で指定があればそちらを優先(run_all.pyからの自動実行用)。
# 引数なしで実行した場合は、下のデフォルト値がそのまま使われる(単体テスト用)。
IMAGE_FOLDER = sys.argv[1] if len(sys.argv) > 1 else r"data\captures\hitball1\camera_b"
OUTPUT_FOLDER = sys.argv[2] if len(sys.argv) > 2 else r"results\20260918\ball_detection\hitball1_camera_b"

# ボール検出のフィルタ条件(Contourベース)
MIN_AREA = 100
MAX_AREA = 4000
MIN_CIRCULARITY = 0.3
MIN_BRIGHTNESS = 150

# ボールが映っている実際の範囲(まだ分かっていないので、いったんNoneにして
# 正検出率の計算をスキップする。目視で範囲が分かったら数値を入れて再実行してもよい)
BALL_START = None
BALL_END = None

# ============================================================
# ファイル名パターン: 000000_bid975_ts3765738517950.bmp
#   先頭: 保存連番、bid: カメラ内部のフレーム連番、ts: カメラ内部のタイムスタンプ
# ============================================================
FRAME_NUM_PATTERN = re.compile(r"^(\d+)_")
BID_TS_PATTERN = re.compile(r"bid(\d+)_ts(\d+)", re.IGNORECASE)


def parse_filename(path):
    """ファイル名から (frame_num, bid, ts) を取り出す。"""
    name = os.path.basename(path)
    frame_num = int(FRAME_NUM_PATTERN.search(name).group(1))
    m = BID_TS_PATTERN.search(name)
    bid = int(m.group(1))
    ts = int(m.group(2))
    return frame_num, bid, ts


# ============================================================
# 画像の読み込み(ファイルパス一覧のみ、実データはまだ読まない)
# ============================================================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.bmp")),
    key=lambda x: parse_filename(x)[0]
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
ball_positions = []      # (frame_num, bid, ts, bx, by, area, circularity) 全候補
best_detections = []     # フレームごとに面積最大の1候補だけ(後工程で使う本命データ)
area_list = []

# 処理時間計測用
time_imread = 0.0
time_diff = 0.0
time_binarize = 0.0
time_contour = 0.0

start_time = time.time()

frame_buffer = {}

t_read0 = time.time()
frame_buffer[0] = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
frame_buffer[1] = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)
t_read1 = time.time()
time_imread += (t_read1 - t_read0)

for i in range(1, len(image_paths) - 1):
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

    frame_num, bid, ts = parse_filename(image_paths[i])

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

    debug_path = os.path.join(DEBUG_FOLDER, f"diff_{frame_num:06d}.jpg")
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

    # ---- 検出結果を記録(複数候補があれば面積最大を「本命」とする) ----
    if len(candidates) > 0:
        for cx, cy, area, circularity, brightness in candidates:
            bx, by = int(cx), int(cy)
            ball_positions.append((frame_num, bid, ts, bx, by, area, circularity))
            area_list.append(area)

        best = max(candidates, key=lambda c: c[2])  # 面積最大を採用
        best_detections.append({
            "frame_num": frame_num,
            "bid": bid,
            "ts": ts,
            "u": best[0],
            "v": best[1],
            "area": best[2],
            "circularity": best[3],
            "num_candidates": len(candidates),
        })

        print(f"フレーム {frame_num:06d} (bid={bid}): {len(candidates)}件検出 " +
            ", ".join([f"({int(c[0])},{int(c[1])},面積={c[2]:.1f})" for c in candidates]))

    # ---- 結果画像の保存(検出があったフレームだけ) ----
    if len(candidates) > 0:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for cx, cy, area, circularity, brightness in candidates:
            bx, by = int(cx), int(cy)
            cv2.circle(vis, (bx, by), 15, (0, 0, 255), 2)
            cv2.putText(vis, f"({bx},{by})", (bx + 10, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(vis, f"Frame: {frame_num:06d} bid={bid}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out_path = os.path.join(OUTPUT_FOLDER, f"result_{frame_num:06d}.jpg")
        cv2.imwrite(out_path, vis)

    if (i - 1) in frame_buffer:
        del frame_buffer[i - 1]

end_time = time.time()
elapsed = end_time - start_time

# ============================================================
# 検出結果をJSONに保存(後工程の三角測量・カメラ間対応付けで使う)
# ============================================================
detections_path = os.path.join(OUTPUT_FOLDER, "detections.json")
with open(detections_path, "w", encoding="utf-8") as f:
    json.dump(best_detections, f, ensure_ascii=False, indent=2)
print(f"\n検出結果を保存しました: {detections_path}")

# ============================================================
# 結果サマリー
# ============================================================
print(f"\n--- 結果サマリー ---")
print(f"総フレーム数: {len(image_paths)}")

detected_frames = sorted(set(pos[0] for pos in ball_positions))
print(f"検出できたフレーム数（ユニーク）: {len(detected_frames)}")
print(f"検出された候補の総数（複数候補含む）: {len(ball_positions)}")
if detected_frames:
    print(f"検出されたフレーム番号: {detected_frames}")

if BALL_START is not None and BALL_END is not None:
    ball_total = BALL_END - BALL_START + 1
    ball_range_frames = set(f for f in detected_frames if BALL_START <= f <= BALL_END)
    false_positive_frames = set(f for f in detected_frames if f < BALL_START or f > BALL_END)
    print(f"ボールあり範囲({BALL_START}-{BALL_END})での検出フレーム数: {len(ball_range_frames)} / {ball_total}フレーム")
    print(f"ボールなし範囲での誤検出フレーム数: {len(false_positive_frames)}")
    print(f"正検出率: {len(ball_range_frames)/ball_total*100:.1f}%")
else:
    print("(BALL_START/BALL_ENDが未設定のため、正検出率の計算はスキップしました。)")

if area_list:
    print(f"検出できた面積: 最小={min(area_list):.1f}, 最大={max(area_list):.1f}, 平均={sum(area_list)/len(area_list):.1f}")

print(f"\n総処理時間: {elapsed:.3f}秒")
print(f"1フレームあたり平均: {elapsed/len(image_paths)*1000:.2f}ms")