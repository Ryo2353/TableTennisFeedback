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
TEMPLATE_PATH = r"ball_template.jpg"
OUTPUT_FOLDER = r"output_detect_template"

MATCH_THRESHOLD = 0.6
SCALES = np.linspace(0.5, 2.0, 16)

SEARCH_MARGIN = 150   # 前フレーム座標からの探索範囲(片側マージン、px)
LOST_LIMIT = 5         # 何フレーム連続で見失ったら全体探索に戻すか

# ============================================================
# テンプレート読み込み
# ============================================================
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
th, tw = template.shape[:2]
print(f"テンプレートサイズ: {tw}x{th}")

# ============================================================
# 画像の読み込み
# ============================================================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")),
    key=lambda x: int(re.search(r'_(\d+)\.jpg$', x).group(1))
)
print(f"画像枚数: {len(image_paths)}")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 検出ループ(前フレーム位置ベースの探索範囲限定つき)
# ============================================================
ball_positions = []
score_list = []

last_position = None   # 直前に検出できた座標(bx, by)
lost_count = 0          # 連続未検出フレーム数

start_time = time.time()

for path in image_paths:
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        continue

    frame_num = int(re.search(r'_(\d+)\.jpg$', path).group(1))

    # ---- 探索範囲の決定 ----
    if last_position is not None and lost_count < LOST_LIMIT:
        lx, ly = last_position
        x1 = max(0, lx - SEARCH_MARGIN)
        y1 = max(0, ly - SEARCH_MARGIN)
        x2 = min(frame.shape[1], lx + SEARCH_MARGIN)
        y2 = min(frame.shape[0], ly + SEARCH_MARGIN)
        search_area = frame[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        # 初回、または見失って一定フレーム経過した場合は全体探索
        search_area = frame
        offset_x, offset_y = 0, 0

    best_score = -1
    best_loc = None
    best_size = None

    for scale in SCALES:
        resized_w = int(tw * scale)
        resized_h = int(th * scale)
        if resized_w < 5 or resized_h < 5:
            continue
        if resized_w > search_area.shape[1] or resized_h > search_area.shape[0]:
            continue

        resized_template = cv2.resize(template, (resized_w, resized_h))
        result = cv2.matchTemplate(search_area, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = (max_loc[0] + offset_x, max_loc[1] + offset_y)  # 元画像座標に変換
            best_size = (resized_w, resized_h)

    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if best_score >= MATCH_THRESHOLD:
        bx = best_loc[0] + best_size[0] // 2
        by = best_loc[1] + best_size[1] // 2
        ball_positions.append((frame_num, bx, by))
        score_list.append(best_score)

        last_position = (bx, by)   # 次フレームの探索範囲の中心として使う
        lost_count = 0

        print(f"フレーム {frame_num:04d}: 検出 ({bx},{by}), スコア={best_score:.3f}, "
              f"探索範囲={'限定' if offset_x or offset_y else '全体'}")
    else:
        lost_count += 1
        print(f"フレーム {frame_num:04d}: 未検出(最高スコア={best_score:.3f}), "
              f"連続未検出={lost_count}")

end_time = time.time()
elapsed = end_time - start_time

# ============================================================
# 結果サマリー
# ============================================================
print(f"\n--- 結果サマリー(テンプレートマッチング + 探索範囲限定) ---")
print(f"総フレーム数: {len(image_paths)}")

BALL_START = 244
BALL_END = 261
ball_total = BALL_END - BALL_START + 1

detected_frames = set(pos[0] for pos in ball_positions)
ball_range_frames = set(f for f in detected_frames if BALL_START <= f <= BALL_END)
false_positive_frames = set(f for f in detected_frames if f < BALL_START or f > BALL_END)

print(f"検出できたフレーム数: {len(detected_frames)}")
print(f"ボールあり範囲({BALL_START}-{BALL_END})での検出数: {len(ball_range_frames)} / {ball_total}フレーム")
print(f"ボールなし範囲での誤検出数: {len(false_positive_frames)}")
print(f"正検出率: {len(ball_range_frames)/ball_total*100:.1f}%")

if score_list:
    print(f"マッチスコア: 最小={min(score_list):.3f}, 最大={max(score_list):.3f}, 平均={sum(score_list)/len(score_list):.3f}")

print(f"\n総処理時間: {elapsed:.3f}秒")
print(f"1フレームあたり平均: {elapsed/len(image_paths)*1000:.2f}ms")