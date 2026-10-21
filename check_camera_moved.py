# -*- coding: utf-8 -*-
"""
撮影の最初と最後に撮った「何もない台」の画像を比べて、
途中でカメラが動いていないかを確かめる。

2枚の画像のずれ(画素)を、位相限定相関(cv2.phaseCorrelate)で求める。
カメラが動いていなければ、ずれはほぼ0(1画素未満)になる。

使い方:
    python check_camera_moved.py <最初の画像> <最後の画像>

    例(カメラAとBで1回ずつ):
    python check_camera_moved.py data\\captures\\20261016_183718\\camera_a\\snapshot_00033_....png data\\captures\\20261016_19xxxx\\camera_a\\snapshot_00001_....png
    python check_camera_moved.py data\\captures\\20261016_183718\\camera_b\\snapshot_00033_....png data\\captures\\20261016_19xxxx\\camera_b\\snapshot_00001_....png

目安:
    ずれ 1画素未満        → 動いていない
    ずれ 1〜3画素         → ほぼ動いていない(クリックの誤差と同じくらい)
    ずれ 3画素より大きい  → 途中でカメラか台が動いた可能性が高い
"""

import sys

import cv2
import numpy as np


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[エラー] 画像を読めません: {path}")
        sys.exit(1)
    return img.astype(np.float32)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    a = load_gray(sys.argv[1])
    b = load_gray(sys.argv[2])
    if a.shape != b.shape:
        print(f"[エラー] 画像の大きさが違います: {a.shape} と {b.shape}")
        sys.exit(1)

    # 明るさの違いの影響を減らすため、ぼかしてから端を弱める窓をかける
    a = cv2.GaussianBlur(a, (5, 5), 0)
    b = cv2.GaussianBlur(b, (5, 5), 0)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    shift = float(np.hypot(dx, dy))

    # 参考: 明るさの差の平均(人や物が写り込んでいないかの目安)
    diff = float(np.mean(np.abs(a - b)))

    print(f"ずれ: 横 {dx:+.2f}画素, 縦 {dy:+.2f}画素  (合計 {shift:.2f}画素)")
    print(f"一致の度合い(1に近いほど同じ見た目): {response:.2f}")
    print(f"明るさの差の平均(参考): {diff:.1f}")
    if response < 0.1:
        print("→ 判定できません(画像の見た目が大きく違う。別の場面の画像か、明るさが大きく変わった可能性)")
    elif shift < 1.0:
        print("→ 動いていない")
    elif shift <= 3.0:
        print("→ ほぼ動いていない(クリックの誤差と同じくらい)")
    else:
        print("→ 途中でカメラか台が動いた可能性が高い")


if __name__ == "__main__":
    main()