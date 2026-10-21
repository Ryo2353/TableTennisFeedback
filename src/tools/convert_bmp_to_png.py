# -*- coding: utf-8 -*-
r"""
撮影フォルダの bmp 画像を png に変換して、容量を減らす。

png は画像を完全に保ったまま圧縮する形式なので、画像の中身(画素の値)は1つも変わらない。
大きさはだいたい3分の1になる。

安全のため、1枚ずつ次の順で処理する:
  1. bmp を読み込んで、同じ名前の png として保存する
  2. 保存した png を読み直し、元の bmp と画素が完全に同じか確かめる
  3. 同じだと確かめられたときだけ、bmp を消す
  (違っていた・読めなかったときは、bmp を残して png の方を消し、最後に一覧を表示する)

使い方(プロジェクトのいちばん上のフォルダで):
    python src\tools\convert_bmp_to_png.py data\captures\hitball1 data\captures\hitball2

    --dry-run を付けると、変換はせずに、対象の枚数と容量だけ表示する
    python src\tools\convert_bmp_to_png.py data\captures\hitball1 --dry-run
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np


def convert_one(bmp_path):
    """1枚変換する。(結果, 元の大きさ, 変換後の大きさ, パス) を返す"""
    png_path = os.path.splitext(bmp_path)[0] + ".png"
    try:
        size_before = os.path.getsize(bmp_path)
        img = cv2.imread(bmp_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return ("読めない", size_before, 0, bmp_path)
        if not cv2.imwrite(png_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            return ("保存失敗", size_before, 0, bmp_path)
        back = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if back is None or back.shape != img.shape or not np.array_equal(back, img):
            if os.path.exists(png_path):
                os.remove(png_path)
            return ("一致しない", size_before, 0, bmp_path)
        size_after = os.path.getsize(png_path)
        os.remove(bmp_path)
        return ("OK", size_before, size_after, bmp_path)
    except Exception as e:  # noqa: BLE001
        return (f"エラー: {e}", 0, 0, bmp_path)


def find_bmps(folders):
    paths = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"[注意] フォルダが見つかりません: {folder}")
            continue
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".bmp"):
                    paths.append(os.path.join(root, f))
    return sorted(paths)


def gb(n):
    return f"{n / 1024 ** 3:.2f}GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="変換するフォルダ(中のフォルダも含めて全部の bmp が対象)")
    ap.add_argument("--dry-run", action="store_true", help="変換せず、対象の枚数と容量だけ表示する")
    ap.add_argument("--workers", type=int, default=4, help="同時に変換する数(既定 4)")
    args = ap.parse_args()

    paths = find_bmps(args.folders)
    total = sum(os.path.getsize(p) for p in paths)
    print(f"対象の bmp: {len(paths)} 枚、合計 {gb(total)}")
    if args.dry_run or not paths:
        return

    t0 = time.time()
    ok = before = after = 0
    failed = []
    with Pool(args.workers) as pool:
        for i, (res, b, a, p) in enumerate(pool.imap_unordered(convert_one, paths, chunksize=8), 1):
            if res == "OK":
                ok += 1
                before += b
                after += a
            else:
                failed.append((res, p))
            if i % 200 == 0 or i == len(paths):
                el = time.time() - t0
                print(f"  {i}/{len(paths)} 枚  ({el:.0f}秒、残り約{el / i * (len(paths) - i):.0f}秒)")

    print("\n--- まとめ ---")
    print(f"変換できた: {ok} / {len(paths)} 枚")
    print(f"容量: {gb(before)} → {gb(after)}  (空いた容量 {gb(before - after)})")
    if failed:
        print(f"\n変換できず bmp のまま残したもの: {len(failed)} 枚")
        for res, p in failed[:20]:
            print(f"  {res}: {p}")
        if len(failed) > 20:
            print(f"  ...ほか {len(failed) - 20} 枚")
    else:
        print("失敗はありません。")


if __name__ == "__main__":
    sys.exit(main())