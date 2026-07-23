"""
卓球台キャリブレーションツール (Python版)
========================================

真上カメラで撮った画像のピクセル座標 → 実世界座標(mm) への
ホモグラフィ変換行列を、画像をクリックして求めるためのスクリプト。

ブラウザ版のツールと同じ考え方(クリックで基準点を登録 → 実座標を入力 →
cv2.findHomography() で行列を計算)をPythonに移植したもの。
計算した行列はJSONファイルに保存し、以後の連番画像の処理(本パイプライン側)
から load_homography() / pixel_to_real() を呼び出して使う想定。

必要なライブラリ:
    pip install opencv-python numpy matplotlib

使い方 (コマンドライン)
------------------------
# 1. 新しく基準点をクリックして行列を計算する
python calibrate_homography.py calibrate --image path/to/calib.jpg --output homography.json

# 2. 保存済みの行列を使って、画像をクリックした点の実座標を確認する(検算用)
python calibrate_homography.py test --image path/to/any_frame.jpg --matrix homography.json

使い方 (他のスクリプトから呼び出す場合)
----------------------------------------
from calibrate_homography import load_homography, pixel_to_real

H, meta = load_homography("homography.json")
X, Y = pixel_to_real(H, 512, 300)
print(X, Y)  # 実世界座標(mm)
"""

import argparse
import json
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 環境によっては "Qt5Agg" 等に変更してください
import matplotlib.pyplot as plt

# 日本語表示のためのフォント設定(文字化け対策)
# 環境にインストールされている日本語フォントを順番に探して使う
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Yu Gothic", "MS Gothic", "Meiryo",       # Windows
    "Hiragino Sans", "Hiragino Kaku Gothic ProN",  # macOS
    "IPAexGothic", "Noto Sans CJK JP",        # Linux
    "DejaVu Sans",                            # 見つからない場合の最終フォールバック
]
plt.rcParams["axes.unicode_minus"] = False


class PointPicker:
    """画像上をクリックして基準点(ピクセル座標)を集めるクラス。

    左クリック: 点を追加
    右クリック: 直前に追加した点を削除
    ウィンドウを閉じる: 入力終了
    """

    def __init__(self, image_rgb):
        self.points = []  # [(x, y), ...]
        self.markers = []  # 描画したマーカー(削除用に保持)
        self.fig, self.ax = plt.subplots(figsize=(10, 7.5))
        self.ax.imshow(image_rgb)
        self.ax.set_title(
            "左クリック:点を追加 / 右クリック:直前の点を削除 / 閉じる:終了"
        )
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        plt.tight_layout()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == 1:
            x, y = event.xdata, event.ydata
            self.points.append((x, y))
            idx = len(self.points)
            marker = self.ax.plot(
                x, y, "+", color="cyan", markersize=12, markeredgewidth=2
            )[0]
            label = self.ax.text(
                x + 8, y - 8, str(idx), color="white", fontsize=11,
                fontweight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )
            self.markers.append((marker, label))
            print(f"[{idx}] px = ({x:.1f}, {y:.1f})")
        elif event.button == 3:
            if self.points:
                self.points.pop()
                marker, label = self.markers.pop()
                marker.remove()
                label.remove()
                print("直前の点を削除しました")
        self.fig.canvas.draw()

    def run(self):
        plt.show()
        return self.points


def pick_points(image_path):
    """画像を表示し、クリックされたピクセル座標のリストを返す"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    picker = PointPicker(img_rgb)
    points = picker.run()
    image_size = [img.shape[1], img.shape[0]]  # [width, height]
    return points, image_size


def prompt_real_coords(pixel_points):
    """各ピクセル点について、実座標(mm)をターミナルで入力してもらう"""
    real_points = []
    print("\n各点の実座標(mm)を入力してください。")
    for i, (px, py) in enumerate(pixel_points, start=1):
        print(f"\n--- 点 {i}  (px: {px:.1f}, {py:.1f}) ---")
        while True:
            try:
                rx = float(input("  実X (mm): "))
                ry = float(input("  実Y (mm): "))
                break
            except ValueError:
                print("  数値で入力してください。")
        real_points.append((rx, ry))
    return real_points


def compute_homography(pixel_points, real_points):
    """対応点からホモグラフィ行列(3x3)を計算する"""
    if len(pixel_points) < 4:
        raise ValueError("基準点は4点以上必要です")
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(real_points, dtype=np.float32)
    # method=0: 全点を使った最小二乗解(点が多いほど誤差が均される)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("行列の計算に失敗しました。点の配置を確認してください。")
    return H


def reprojection_error(H, pixel_points, real_points):
    """計算した行列で各点を逆算し、実測値とのズレ(mm)を返す"""
    src = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(real_points, dtype=np.float32)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    return rms, errors


def pixel_to_real(H, x, y):
    """1点のピクセル座標を実座標(mm)に変換する(パイプラインから呼ぶ用)"""
    pt = np.array([[[x, y]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0, 0, 0]), float(result[0, 0, 1])


def save_homography(path, H, pixel_points, real_points, image_size, note=""):
    data = {
        "note": note,
        "image_size": image_size,
        "homography_matrix": H.tolist(),
        "points": [
            {"px": list(p), "real": list(r)}
            for p, r in zip(pixel_points, real_points)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n保存しました: {path}")


def load_homography(path):
    """保存済みのJSONから行列を読み込む。戻り値: (H, メタデータdict)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float64)
    return H, data


def run_calibrate(args):
    print(f"画像を読み込みます: {args.image}")
    pixel_points, image_size = pick_points(args.image)

    if len(pixel_points) < 4:
        print(f"エラー: 基準点は4点以上必要です(現在 {len(pixel_points)} 点)")
        sys.exit(1)

    real_points = prompt_real_coords(pixel_points)
    H = compute_homography(pixel_points, real_points)
    rms, errors = reprojection_error(H, pixel_points, real_points)

    print("\n=== 計算結果 ===")
    print(H)
    print(f"\n再投影誤差 RMS: {rms:.2f} mm")
    for i, e in enumerate(errors, start=1):
        print(f"  点{i}: 誤差 {e:.2f} mm")

    note = input("\nメモ(原点の定義など。空欄可): ")
    save_homography(args.output, H, pixel_points, real_points, image_size, note)


def run_test(args):
    H, data = load_homography(args.matrix)
    print(f"読み込んだ行列のメモ: {data.get('note', '(なし)')}")

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(img_rgb)
    ax.set_title("クリックすると実座標を表示します(閉じて終了)")

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        X, Y = pixel_to_real(H, x, y)
        print(f"px: ({x:.1f}, {y:.1f})  ->  実座標: X = {X:.1f} mm, Y = {Y:.1f} mm")
        ax.plot(x, y, "x", color="orange", markersize=10, markeredgewidth=2)
        ax.text(x + 8, y - 8, f"({X:.0f}, {Y:.0f})", color="orange", fontsize=9)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="卓球台キャリブレーションツール")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="基準点をクリックして新しい行列を計算する")
    p_cal.add_argument("--image", required=True, help="キャリブレーション用画像のパス")
    p_cal.add_argument("--output", default="homography.json", help="保存先JSONファイル")
    p_cal.set_defaults(func=run_calibrate)

    p_test = sub.add_parser("test", help="保存済みの行列で変換テストする")
    p_test.add_argument("--image", required=True, help="テストしたい画像のパス")
    p_test.add_argument("--matrix", required=True, help="保存済みの行列JSONファイル")
    p_test.set_defaults(func=run_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()