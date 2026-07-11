import cv2
import numpy as np
import sys

def analyze(path):
    binary = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if binary is None:
        print(f"読み込み失敗: {path}")
        return

    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\n=== {path} ===")
    if len(contours) == 0:
        print("  輪郭が見つかりません")
        return

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        print(f"  面積={area:.1f}, 周囲長={perimeter:.1f}, 円形度={circularity:.3f}, 凸性={convexity:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python check_contours.py <画像パス> [画像パス2] ...")
        sys.exit(1)

    for path in sys.argv[1:]:
        analyze(path)