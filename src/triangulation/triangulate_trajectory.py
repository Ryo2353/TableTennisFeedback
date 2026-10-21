"""
Detectball_frame.py が出力した detections.json (カメラA用・カメラB用) を、
offline_sync_check.py と同じ考え方(block_id基準のdelta)でカメラA/B間の
フレームを対応付け、対応が取れたフレームごとに三角測量して卓球台基準の
3D軌跡(X, Y, Z の時系列)を出力する。Phase B-3(バウンドタイミング推定)の
入力データを作るためのスクリプト。

なぜタイムスタンプ(ts)そのものではなくblock_id(bid)のdeltaを使うか:
    カメラAとカメラBのtsは、それぞれのカメラが独立に持つ時計に基づいており、
    ドリフト(実機検証で10秒あたり約78ms)がある。一方、両カメラは同じ
    録画ループから同時に撮影開始しているはずなので、「録画開始から何コマ目か」
    (= bid - そのカメラの最初のbid)は、多少のフレーム落ちがあっても
    両カメラでほぼ一致する。そのためdelta基準でペアリングする。

使い方は2通り:

【新】撮りながら検出(capture_live_detect.py)の結果を使う場合 ← 10/16〜
    python triangulate_trajectory.py <録画フォルダ(rec01など)> <stereo_result.npz> <table_transform.npz> <出力csv>

    録画フォルダの中の camera_a/detections.json と camera_b/detections.json を読む。
    画像は保存していないので、各検出に入っている "delta"(= bid - 録画開始時のbid)を
    そのまま使って対応付ける。

    例:
    python triangulate_trajectory.py \\
        data\\captures\\20261016_190000\\rec01 \\
        results\\20261016\\stereo_result.npz results\\20261016\\table_transform.npz \\
        results\\20261016\\rec01_trajectory.csv

【旧】保存した画像フォルダ + Detectball_frame.py の結果を使う場合
    python triangulate_trajectory.py \
        <カメラA全体フォルダ> <カメラB全体フォルダ> \
        <カメラAのdetections.json> <カメラBのdetections.json> \
        <stereo_result.npz> <table_transform.npz> \
        <出力csv>

例:
    python triangulate_trajectory.py \
        data\\captures\\naka_oku\\camera_a data\\captures\\naka_oku\\camera_b \
        results\\20260918\\ball_detection\\naka_oku_camera_a\\detections.json \
        results\\20260918\\ball_detection\\naka_oku_camera_b\\detections.json \
        results\\20260918\\stereo_result.npz results\\20260918\\table_transform.npz \
        results\\20260918\\ball_detection\\naka_oku_trajectory.csv
"""

import sys
import os
import re
import json
import csv
import numpy as np

from triangulate import load_calibration, build_projection_matrices, triangulate, apply_table_transform

BID_TS_PATTERN = re.compile(r"bid(\d+)_ts(\d+)", re.IGNORECASE)


def get_base_bid(folder):
    """そのカメラのセッションフォルダ全体から、最小のbid(=録画開始時点のbid)を得る。"""
    bids = []
    for fname in os.listdir(folder):
        m = BID_TS_PATTERN.search(fname)
        if m:
            bids.append(int(m.group(1)))
    if not bids:
        raise ValueError(f"{folder} にbidを含むファイルが見つかりません。")
    return min(bids)


def load_detections(path, base_bid):
    """旧: 画像フォルダから求めた基準bidで delta を計算する"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for d in data:
        delta = d["bid"] - base_bid
        result[delta] = d
    return result


def load_detections_with_delta(path):
    """新: capture_live_detect.py が記録した delta をそのまま使う"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for d in data:
        if d.get("delta") is None:
            raise ValueError(f"{path} に delta がありません。capture_live_detect.py の出力か確認してください。")
        result[d["delta"]] = d
    return result


def main():
    if len(sys.argv) == 5:
        # 新: 録画フォルダ(rec01 など)を指定
        rec_dir, stereo_npz_path, table_transform_path, out_csv = sys.argv[1:5]
        det_a_path = os.path.join(rec_dir, "camera_a", "detections.json")
        det_b_path = os.path.join(rec_dir, "camera_b", "detections.json")
        info_path = os.path.join(rec_dir, "session_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            print(f"[情報] カメラA 録画開始時のbid: {info['camera_a']['first_bid']}  "
                  f"コマ数 {info['camera_a']['captured']}  コマ落ち {info['camera_a']['drop_count']}")
            print(f"[情報] カメラB 録画開始時のbid: {info['camera_b']['first_bid']}  "
                  f"コマ数 {info['camera_b']['captured']}  コマ落ち {info['camera_b']['drop_count']}")
        det_a = load_detections_with_delta(det_a_path)
        det_b = load_detections_with_delta(det_b_path)
    elif len(sys.argv) == 8:
        # 旧: 画像フォルダ + detections.json を指定
        folder_a, folder_b, det_a_path, det_b_path, stereo_npz_path, table_transform_path, out_csv = sys.argv[1:8]
        base_a = get_base_bid(folder_a)
        base_b = get_base_bid(folder_b)
        print(f"[情報] カメラA 基準bid: {base_a}")
        print(f"[情報] カメラB 基準bid: {base_b}")
        det_a = load_detections(det_a_path, base_a)
        det_b = load_detections(det_b_path, base_b)
    else:
        print(__doc__)
        sys.exit(1)

    if not det_a or not det_b:
        print("[エラー] どちらかのカメラで検出が1つもありません。")
        sys.exit(1)
    print(f"[情報] カメラA 検出フレーム数: {len(det_a)} (delta範囲: {min(det_a)}〜{max(det_a)})")
    print(f"[情報] カメラB 検出フレーム数: {len(det_b)} (delta範囲: {min(det_b)}〜{max(det_b)})")

    common_deltas = sorted(set(det_a.keys()) & set(det_b.keys()))
    print(f"\n[情報] deltaが一致して対応付けられたフレーム数: {len(common_deltas)}")
    if not common_deltas:
        print("[エラー] 対応が取れたフレームが1つもありません。"
              "bidの飛び方(フレーム落ちのタイミング)がカメラA/Bで大きく違う可能性があります。"
              "±1程度のずれを許容する対応付けを検討してください。")
        sys.exit(1)

    K1, dist1, K2, dist2, R, T = load_calibration(stereo_npz_path)
    P1, P2 = build_projection_matrices(K1, K2, R, T)

    pts1 = np.array([[det_a[d]["u"], det_a[d]["v"]] for d in common_deltas], dtype=np.float64)
    pts2 = np.array([[det_b[d]["u"], det_b[d]["v"]] for d in common_deltas], dtype=np.float64)

    estimated_mm = triangulate(P1, P2, K1, dist1, K2, dist2, pts1, pts2)

    td = np.load(table_transform_path)
    estimated_mm = apply_table_transform(estimated_mm, td["R"], td["t"])

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["delta", "ts_a", "ts_b", "x_mm", "y_mm", "z_mm"])
        for i, d in enumerate(common_deltas):
            x, y, z = estimated_mm[i]
            writer.writerow([d, det_a[d]["ts"], det_b[d]["ts"], f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"])

    print(f"\n保存しました: {out_csv}")
    print("\n=== 軌跡(先頭10点) ===")
    for i, d in enumerate(common_deltas[:10]):
        x, y, z = estimated_mm[i]
        print(f"  delta={d:5d}  X={x:8.1f}  Y={y:8.1f}  Z={z:8.1f}")


if __name__ == "__main__":
    main()