"""
ボール検出の速度測定(「検出そのもの」の時間だけを測る)

元の Detectball_frame.py の「1フレームあたり平均」には、画像の読み込み・
デバッグ画像の書き出し(毎フレーム)・結果画像の書き出し・print が
すべて含まれていた。撮りながら検出する方式(D)が可能かを判断するには、
それらを除いた「検出そのもの」の時間を知る必要がある。

このスクリプトは:
  1. 画像を先にメモリへ読み込み(読み込み時間は別に計測)
  2. 次の方式それぞれで、検出の各段階の時間を測る
       - 元の方式(フル解像度・3フレーム差分)
       - 縮小1/2(3フレーム差分)
       - 縮小1/2 + 背景差分
       - フル解像度 + 背景差分
  3. 元の方式と比べて、同じフレームで同じ位置に検出できたかを数える
画像の書き出しは一切しない。

使い方:
    python benchmark_detect.py <画像フォルダ> [--max-frames 600] [--roi x,y,w,h]

例:
    python benchmark_detect.py data\\captures\\hitball1\\camera_a
    python benchmark_detect.py data\\captures\\hitball1\\camera_a --roi 0,0,1440,400

測定のときは、ノートPCを電源につなぎ、高パフォーマンスモードにすること。
"""

import argparse
import glob
import os
import re
import time

import cv2
import numpy as np

# ---- 検出条件(Detectball_frame.py と同じ) ----
MIN_AREA = 100
MAX_AREA = 4000
MIN_CIRCULARITY = 0.3
MIN_BRIGHTNESS = 150
THRESH = 50
KERNEL = np.ones((3, 3), np.uint8)

FRAME_NUM_PATTERN = re.compile(r"^(\d+)_")

# 比べる方式: (表示名, 縮小率, 差分方式)
VARIANTS = [
    ("元の方式(フル・3フレーム差分)", 1.0, "three"),
    ("縮小1/2・3フレーム差分", 0.5, "three"),
    ("縮小1/2・背景差分", 0.5, "bg"),
    ("フル・背景差分", 1.0, "bg"),
]

STAGES = ["前処理", "差分", "二値化+ノイズ", "輪郭", "判定"]


def frame_num(path):
    return int(FRAME_NUM_PATTERN.search(os.path.basename(path)).group(1))


def preprocess(img, roi, scale):
    """ROIの切り出しと縮小(撮りながら検出するなら、各フレーム1回だけ行う処理)"""
    if roi is not None:
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def detect(prev, cur, nxt, bg, method, scale, times):
    """1フレーム分の検出。面積最大の候補 (cx, cy, area) を返す(なければNone)。"""
    t0 = time.perf_counter()
    if method == "three":
        diff = cv2.bitwise_and(cv2.absdiff(prev, cur), cv2.absdiff(cur, nxt))
    else:
        diff = cv2.absdiff(cur, bg)
    t1 = time.perf_counter()

    _, binary = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, KERNEL, iterations=2)
    binary = cv2.erode(binary, KERNEL, iterations=1)
    t2 = time.perf_counter()

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    t3 = time.perf_counter()

    # 縮小した分だけ、面積のしきい値と明るさチェックの範囲も小さくする
    s2 = scale * scale
    r = max(2, int(round(5 * scale)))
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA * s2 or area > MAX_AREA * s2:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        if 4 * np.pi * area / perimeter ** 2 < MIN_CIRCULARITY:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        patch = cur[max(0, int(cy) - r):int(cy) + r, max(0, int(cx) - r):int(cx) + r]
        if patch.size == 0 or patch.mean() < MIN_BRIGHTNESS:
            continue
        if best is None or area > best[2]:
            best = (cx, cy, area)
    t4 = time.perf_counter()

    times["差分"] += t1 - t0
    times["二値化+ノイズ"] += t2 - t1
    times["輪郭"] += t3 - t2
    times["判定"] += t4 - t3
    return best


def run_variant(raw_frames, roi, scale, method):
    times = {s: 0.0 for s in STAGES}

    t0 = time.perf_counter()
    frames = [preprocess(f, roi, scale) for f in raw_frames]
    times["前処理"] = time.perf_counter() - t0

    # 背景画像: 最初の30フレームの中央値(1回だけ作るので時間には含めない)
    bg = None
    if method == "bg":
        bg = np.median(np.stack(frames[:30]), axis=0).astype(np.uint8)

    ox, oy = (roi[0], roi[1]) if roi is not None else (0, 0)
    results = {}
    for i in range(1, len(frames) - 1):
        best = detect(frames[i - 1], frames[i], frames[i + 1], bg, method, scale, times)
        if best is not None:
            # 元画像(フル解像度)の座標に戻して記録
            results[i] = (best[0] / scale + ox, best[1] / scale + oy)

    n = len(frames) - 2
    per_frame = {s: times[s] / n * 1000 for s in STAGES}
    # 前処理は全フレーム分なのでフレーム数で割り直す
    per_frame["前処理"] = times["前処理"] / len(frames) * 1000
    per_frame["合計"] = sum(per_frame[s] for s in STAGES)
    return per_frame, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--roi", type=str, default=None, help="x,y,w,h(元画像の座標)")
    args = parser.parse_args()

    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None

    folder_abs = os.path.abspath(args.folder)
    paths = sorted(glob.glob(os.path.join(args.folder, "*.bmp")), key=frame_num)
    paths = paths[:args.max_frames]
    if len(paths) < 40:
        print("画像が少なすぎます(40枚以上必要)。")
        print(f"  探したフォルダ: {folder_abs}")
        print(f"  フォルダが存在するか: {os.path.isdir(folder_abs)}")
        print(f"  見つかった .bmp の数: {len(paths)}")
        if os.path.isdir(folder_abs):
            print(f"  フォルダ内の先頭5件: {sorted(os.listdir(folder_abs))[:5]}")
        print(f"  今いるフォルダ(カレントディレクトリ): {os.getcwd()}")
        return

    # ---- 読み込み(別に計測) ----
    t0 = time.perf_counter()
    raw = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
    load_ms = (time.perf_counter() - t0) / len(paths) * 1000

    print(f"フォルダ: {args.folder}")
    print(f"フレーム数: {len(raw)}  画像サイズ: {raw[0].shape[1]}x{raw[0].shape[0]}")
    print(f"ROI: {roi if roi else 'なし(全体)'}")
    print(f"OpenCVのスレッド数: {cv2.getNumThreads()}")
    print(f"\n読み込み(参考・撮りながら検出なら不要): {load_ms:.2f} ms/コマ\n")

    header = f"{'方式':<26}" + "".join(f"{s:>12}" for s in STAGES) + f"{'合計':>10}{'検出コマ':>10}{'元と一致':>10}"
    print(header)
    print("-" * len(header))

    baseline = None
    for name, scale, method in VARIANTS:
        per_frame, results = run_variant(raw, roi, scale, method)
        if baseline is None:
            baseline = results
        match = sum(
            1 for i, (u, v) in results.items()
            if i in baseline and abs(u - baseline[i][0]) <= 10 and abs(v - baseline[i][1]) <= 10
        )
        line = f"{name:<26}" + "".join(f"{per_frame[s]:>12.2f}" for s in STAGES)
        line += f"{per_frame['合計']:>10.2f}{len(results):>10}{match:>10}"
        print(line)

    print("\n(時間の単位はすべて ms/コマ)")
    print("目安: 371fpsなら 2.7ms/コマ以内、200fpsなら 5.0ms/コマ以内(カメラ1台あたり)")
    print("「元と一致」= 元の方式と同じフレームで、10px以内の位置に検出できた数")


if __name__ == "__main__":
    main()