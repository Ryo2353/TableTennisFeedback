"""
<hitball>_trajectory.csv から、fit_bounce.py の「Zが最小の点+前後1点、
計3点で2次関数を一意に決めて内挿する」という着地点算出の考え方を、
実際のデータでグラフとして見せる(ゼミ発表用)。

注意:
    これは説明用の可視化であり、fit_bounce.py本体のロジック(delta差が
    大きいところでの打球の自動分割など)を完全に再現したものではない。
    1つのhitball_trajectory.csvの中に、着地(バウンド)が1回だけ
    含まれているデータを使うことを想定している。

使い方:
    python plot_bounce_fit.py <hitball名>_trajectory.csv [出力png]

例:
    python plot_bounce_fit.py results\\20260918\\ball_detection\\hitball1_trajectory.csv results\\20260918\\bounce_fit_plot.png
"""

import sys
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

_JP_FONT_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_jp_font = next((f for f in _JP_FONT_CANDIDATES if f in _available), None)
if _jp_font:
    matplotlib.rcParams["font.family"] = _jp_font
else:
    print("[警告] 日本語フォントが見つからなかったため、ラベルを英語表記にします。")
matplotlib.rcParams["axes.unicode_minus"] = False


def fit_vertex_3pts(t3, z3):
    """3点を通る2次関数を一意に決め、頂点の(t, z)を返す。"""
    a, b, c = np.polyfit(t3, z3, 2)
    if a == 0:
        # 3点がほぼ直線上(通常起きないが念のため)
        i_min = int(np.argmin(z3))
        return t3[i_min], z3[i_min], (a, b, c)
    t_vertex = -b / (2 * a)
    z_vertex = a * t_vertex ** 2 + b * t_vertex + c
    return t_vertex, z_vertex, (a, b, c)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.rsplit(".csv", 1)[0] + "_bounce_fit_plot.png"

    delta, z = [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delta.append(float(row["delta"]))
            z.append(float(row["z_mm"]))

    delta = np.array(delta)
    z = np.array(z)
    n = len(delta)
    print(f"読み込んだ点数: {n}")
    if n < 3:
        print("[エラー] 3点以上必要です。")
        sys.exit(1)

    min_i = int(np.argmin(z))
    if min_i == 0 or min_i == n - 1:
        print("[警告] 最下点がデータの端にあるため、前後1点ずつを取れません。"
              "この打球のデータでは可視化に適さない可能性があります。")
        min_i = max(1, min(min_i, n - 2))

    idx3 = [min_i - 1, min_i, min_i + 1]
    t3 = delta[idx3]
    z3 = z[idx3]

    t_vertex, z_vertex, (a, b, c) = fit_vertex_3pts(t3, z3)
    print(f"最下点: delta={delta[min_i]:.1f}, z={z[min_i]:.1f}mm")
    print(f"内挿した頂点: delta={t_vertex:.2f}, z={z_vertex:.2f}mm")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(delta, z, color="#B0B6BE", linewidth=1.5, marker="o", markersize=4,
            label="観測データ全体" if _jp_font else "All observed data", zorder=2)
    ax.scatter(t3, z3, color="#1B2A4A", s=140, zorder=4,
               label="内挿に使う3点(最下点+前後1点)" if _jp_font else "3 points used for interpolation")

    t_fit = np.linspace(t3.min() - 1, t3.max() + 1, 200)
    z_fit = a * t_fit ** 2 + b * t_fit + c
    ax.plot(t_fit, z_fit, color="#3E8E7E", linewidth=2.5,
            label="内挿した2次関数" if _jp_font else "Interpolated parabola", zorder=3)

    ax.scatter([t_vertex], [z_vertex], color="#E4572E", s=220, marker="*", zorder=5,
               label="頂点(=着地点のZ)" if _jp_font else "Vertex (landing point Z)")
    ax.axhline(0, color="black", linewidth=1, linestyle=":")

    ax.set_xlabel("時間経過 (delta)" if _jp_font else "Time (delta)")
    ax.set_ylabel("Z (mm)  ※卓球台からの高さ" if _jp_font else "Z (mm) - height above table")
    ax.set_title("着地点(バウンド位置)の算出方法" if _jp_font else "How the landing point (bounce) is computed",
                  fontsize=15)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"\n保存しました: {out_path}")


if __name__ == "__main__":
    main()