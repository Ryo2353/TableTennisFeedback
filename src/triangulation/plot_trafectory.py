"""
<hitball>_trajectory.csv (triangulate_trajectory.pyの出力) から、
1つの打球の3D軌跡を可視化する(ゼミ発表用)。

出力する2枚組の図:
    左: 卓球台を真上から見た軌跡(X-Y平面)。台の枠・センターラインも重ねる。
    右: 時間(delta)に対する高さ(Z)の変化。バウンドで一度Zが最小になり、
        跳ね返って再び上がる様子が見える。

使い方:
    python plot_trajectory.py <hitball名>_trajectory.csv [出力png]

例:
    python plot_trajectory.py results\\20260918\\ball_detection\\hitball1_trajectory.csv results\\20260918\\trajectory_plot.png
"""

import sys
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_JP_FONT_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_jp_font = next((f for f in _JP_FONT_CANDIDATES if f in _available), None)
if _jp_font:
    matplotlib.rcParams["font.family"] = _jp_font
else:
    print("[警告] 日本語フォントが見つからなかったため、ラベルを英語表記にします。")
matplotlib.rcParams["axes.unicode_minus"] = False

TABLE_WIDTH_X = 1165.0
TABLE_LENGTH_Y = 840.0
CENTERLINE_X = TABLE_WIDTH_X / 2


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.rsplit(".csv", 1)[0] + "_plot.png"

    delta, x, y, z = [], [], [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delta.append(float(row["delta"]))
            x.append(float(row["x_mm"]))
            y.append(float(row["y_mm"]))
            z.append(float(row["z_mm"]))

    delta = np.array(delta)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    n = len(delta)
    print(f"読み込んだ点数: {n}")
    if n == 0:
        print("[エラー] データが1件もありません。")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- 左: 真上から見た軌跡(X-Y) ----
    ax1.add_patch(Rectangle((0, 0), TABLE_WIDTH_X, TABLE_LENGTH_Y,
                             fill=False, edgecolor="black", linewidth=2, zorder=1))
    ax1.axvline(CENTERLINE_X, color="black", linewidth=1.2, linestyle="--", zorder=1,
                label="センターライン" if _jp_font else "Centerline")
    sc1 = ax1.scatter(x, y, c=delta, cmap="viridis", s=40, zorder=3)
    ax1.plot(x, y, color="#999999", linewidth=1, alpha=0.6, zorder=2)
    fig.colorbar(sc1, ax=ax1, shrink=0.8, label="時間経過(delta)" if _jp_font else "Time (delta)")
    ax1.set_xlim(-50, TABLE_WIDTH_X + 50)
    ax1.set_ylim(-50, TABLE_LENGTH_Y + 50)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title("軌跡(真上から見た図)" if _jp_font else "Trajectory (top view)")
    ax1.legend(loc="upper right", fontsize=9)

    # ---- 右: 時間に対する高さ(Z)の変化 ----
    ax2.plot(delta, z, color="#3E8E7E", linewidth=2, marker="o", markersize=4)
    min_i = int(np.argmin(z))
    ax2.scatter([delta[min_i]], [z[min_i]], color="#E4572E", s=140, zorder=5,
                label="最下点(バウンド付近)" if _jp_font else "Lowest point (near bounce)")
    ax2.axhline(0, color="black", linewidth=1, linestyle=":")
    ax2.set_xlabel("時間経過 (delta)" if _jp_font else "Time (delta)")
    ax2.set_ylabel("Z (mm)  ※卓球台からの高さ" if _jp_font else "Z (mm) - height above table")
    ax2.set_title("高さの変化" if _jp_font else "Height over time")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(f"3D軌跡の復元結果 (n={n}点)" if _jp_font else f"Reconstructed 3D trajectory (n={n} points)",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"\n保存しました: {out_path}")


if __name__ == "__main__":
    main()