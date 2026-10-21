# -*- coding: utf-8 -*-
"""
保存済みの連番画像(capture_dual.pyで撮影したもの)から、
カメラ間の同期精度をオフラインで検証するスクリプト。

ファイル名の例: 000000_bid975_ts3765738517950.bmp
  - 数字部分(先頭): 保存時の連番
  - bid の後の数字: block_id(カメラ内部のフレーム連番)
  - ts の後の数字: カメラ内部のハードウェアタイムスタンプ

やっていること:
- camera_a, camera_b それぞれのフォルダから、ファイル名を解析してblock_idとtimestampを取得
- block_idを基準にした「経過フレーム数(delta)」で、両カメラの対応するフレームを正しくペアリング
  (sync_test.pyのオフライン版、リアルタイム取得なしで同じ検証ができる)
- ペアリングできたフレーム同士のタイムスタンプの差を計算し、
  平均・標準偏差・ドリフト(前半と後半でどれだけ変化したか)を出す

使い方:
    python offline_sync_check.py <セッションフォルダのパス>
    (例: python offline_sync_check.py captures/20260805_143012)

セッションフォルダの中に camera_a/ と camera_b/ がある前提です。
"""

import os
import re
import sys

import numpy as np

FILENAME_PATTERN = re.compile(r"bid(\d+)_ts(\d+)", re.IGNORECASE)


def parse_folder(folder):
    """フォルダ内の画像ファイル名から (block_id, timestamp) のリストを作る"""
    entries = []
    for fname in sorted(os.listdir(folder)):
        m = FILENAME_PATTERN.search(fname)
        if m:
            block_id = int(m.group(1))
            ts = int(m.group(2))
            entries.append((block_id, ts, fname))
    return entries


def main():
    if len(sys.argv) < 2:
        print("使い方: python offline_sync_check.py <セッションフォルダのパス>")
        sys.exit(1)

    session_dir = sys.argv[1]
    dir_a = os.path.join(session_dir, "camera_a")
    dir_b = os.path.join(session_dir, "camera_b")

    if not os.path.isdir(dir_a) or not os.path.isdir(dir_b):
        print(f"camera_a / camera_b フォルダが見つかりません: {session_dir}")
        sys.exit(1)

    entries_a = parse_folder(dir_a)
    entries_b = parse_folder(dir_b)

    print(f"カメラA: {len(entries_a)} 枚のファイルを検出")
    print(f"カメラB: {len(entries_b)} 枚のファイルを検出")

    if len(entries_a) < 5 or len(entries_b) < 5:
        print("解析できるファイルが少なすぎます。ファイル名の形式を確認してください。")
        return

    # block_idの連番チェック(コマ落ちの確認、参考情報)
    for label, entries in (("カメラA", entries_a), ("カメラB", entries_b)):
        bids = [e[0] for e in entries]
        gaps = sum(1 for i in range(1, len(bids)) if bids[i] - bids[i - 1] != 1)
        print(f"[{label}] block_idの連番の飛び: {gaps}箇所")

    # block_id基準の delta(経過フレーム数)でペアリング
    base_a = entries_a[0][0]
    base_b = entries_b[0][0]
    delta_to_ts_a = {bid - base_a: ts for bid, ts, _ in entries_a}
    delta_to_ts_b = {bid - base_b: ts for bid, ts, _ in entries_b}

    common_deltas = sorted(set(delta_to_ts_a.keys()) & set(delta_to_ts_b.keys()))
    print(f"\n共通して対応付けられたフレーム数: {len(common_deltas)}")

    if len(common_deltas) < 5:
        print("ペアリングできたフレームが少なすぎます。撮影データを確認してください。")
        return

    ts_a = np.array([delta_to_ts_a[d] for d in common_deltas], dtype=np.int64)
    ts_b = np.array([delta_to_ts_b[d] for d in common_deltas], dtype=np.int64)
    diff = ts_a - ts_b

    print("\n=== カメラ間タイムスタンプの差(A - B) ===")
    print(f"平均: {diff.mean():.1f}  (単位不明。ns想定なら /1e6 でms換算 = {diff.mean()/1e6:.3f} ms)")
    print(f"標準偏差: {diff.std():.1f}  ({diff.std()/1e6:.3f} ms)")
    print(f"最小: {diff.min()}  最大: {diff.max()}")

    half = len(common_deltas) // 2
    first_half_mean = diff[:half].mean()
    second_half_mean = diff[half:].mean()
    drift = second_half_mean - first_half_mean

    print("\n=== ドリフト(時間とともにズレが広がっていないか) ===")
    print(f"前半平均: {first_half_mean:.1f}  後半平均: {second_half_mean:.1f}")
    print(f"変化量: {drift:.1f}  ({drift/1e6:.3f} ms)")

    print("\n=== 判定の目安(フレーム間隔の目安: 227fps換算で約4.4ms=4,400,000ns) ===")
    frame_interval_ns = 1e9 / 227.0
    if abs(drift) < frame_interval_ns and diff.std() < frame_interval_ns * 2:
        print("→ ドリフト・ばらつきともにフレーム間隔程度以下。同期は良好とみられます。")
    else:
        print("→ ドリフトまたはばらつきがフレーム間隔より大きく、同期に問題がある可能性があります。")


if __name__ == "__main__":
    main()