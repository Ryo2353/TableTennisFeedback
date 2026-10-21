"""
all_landings.csv (run_all.py の出力) から、卓球台上の着地点分布を
ヒートマップとして可視化する(Phase C-2)。

考え方:
    着地点の数がまだ少ない(8点)ため、点1つ1つが「ここにピンポイントで
    落ちた」という意味ではなく、周辺にもある程度落ちる可能性がある、と
    ぼかして表現するのが実態に合っている。そこで各点をガウス分布(カーネル)
    としてなめらかに広げてから重ね合わせる KDE(カーネル密度推定)を使う。
    点が少ないうちは「傾向のイメージ」程度のものとして見るのが妥当で、
    データが増えるほど本来の分布に近づいていく。

    色は「密度(多い/少ない)」という大小関係を表すだけなので、単一色相を
    薄い→濃いに変化させるsequentialな配色にする(色相を何色も使う
    レインボー配色は大小関係の直感的な把握を妨げるため避ける)。

使い方:
    python visualize_heatmap.py <all_landings.csvのパス> [出力png]

例:
    python visualize_heatmap.py results\\20260918\\ball_detection\\all_landings.csv results\\20260918\\ball_detection\\landing_heatmap.png
"""

import sys
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- 日本語フォント設定 ----
# Windows環境で標準的に入っているフォントを優先的に探し、見つかったものを使う。
# (これを設定しないと、グラフ中の日本語が全部四角(□)の文字化けになる。)
_JP_FONT_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_jp_font = next((f for f in _JP_FONT_CANDIDATES if f in _available), None)
if _jp_font:
    matplotlib.rcParams["font.family"] = _jp_font
else:
    print("[警告] 日本語フォントが見つからなかったため、ラベルを英語表記にします。")
matplotlib.rcParams["axes.unicode_minus"] = False

# 卓球台のサイズ(mm)。table_transform.npz の座標系に合わせている
# (X: 0〜1165 = 卓球台の幅方向、Y: 0〜840 = 卓球台の長さ方向。
#  center1=582.5,0 center2=582.5,840 がセンターライン=X方向の中央でY方向に伸びる線)。
TABLE_WIDTH_X = 1165.0
TABLE_LENGTH_Y = 840.0
CENTERLINE_X = TABLE_WIDTH_X / 2


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        csv_path.rsplit(".csv", 1)[0].rsplit("all_landings", 1)[0] + "landing_heatmap.png"

    xs, ys, labels = [], [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x_mm"]))
            ys.append(float(row["y_mm"]))
            labels.append(row["hitball"])

    xs = np.array(xs)
    ys = np.array(ys)
    n = len(xs)
    print(f"読み込んだ着地点数: {n}")
    if n == 0:
        print("[エラー] 着地点が1つもありません。")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # ---- 密度(KDE)をグリッド上で計算してヒートマップとして描画 ----
    # 点が少ない(2点未満)とKDEが計算できないので、その場合はスキップして
    # 点(散布図)だけ描画する。
    grid_x, grid_y = np.mgrid[0:TABLE_WIDTH_X:200j, 0:TABLE_LENGTH_Y:200j]

    if n >= 2:
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(np.vstack([xs, ys]))
            density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
            im = ax.pcolormesh(
                grid_x, grid_y, density,
                cmap="YlOrRd",  # sequential: 単一系統色相、薄い→濃い
                shading="gouraud",
                alpha=0.85,
                zorder=1,
            )
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("着地密度(推定・相対値)" if _jp_font else "Estimated landing density (relative)", fontsize=10)
        except Exception as e:
            print(f"[警告] KDE計算に失敗したため、散布図のみ表示します: {e}")
    else:
        print("[情報] 点数が少ないためKDEヒートマップは省略し、散布図のみ表示します。")

    # ---- 卓球台の外枠・センターライン ----
    ax.add_patch(Rectangle((0, 0), TABLE_WIDTH_X, TABLE_LENGTH_Y,
                            fill=False, edgecolor="black", linewidth=2, zorder=3))
    center_label = "センターライン" if _jp_font else "Centerline"
    ax.axvline(CENTERLINE_X, color="black", linewidth=1.5, linestyle="--", zorder=3, label=center_label)

    # ---- 実際の着地点(散布図)を重ねる ----
    points_label = f"着地点 (n={n})" if _jp_font else f"Landing points (n={n})"
    ax.scatter(xs, ys, s=90, facecolor="white", edgecolor="black",
               linewidth=1.5, zorder=4, label=points_label)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, zorder=5)

    ax.set_xlim(-50, TABLE_WIDTH_X + 50)
    ax.set_ylim(-50, TABLE_LENGTH_Y + 50)
    ax.set_aspect("equal")
    if _jp_font:
        ax.set_xlabel("X (mm)  ※卓球台の幅方向(マシン側エンドラインの長さ)", fontsize=10)
        ax.set_ylabel("Y (mm)  ※卓球台の長さ方向", fontsize=10)
        ax.set_title(f"多球練習 着地点分布ヒートマップ (n={n})", fontsize=13)
    else:
        ax.set_xlabel("X (mm) - table width direction (machine-side endline)", fontsize=10)
        ax.set_ylabel("Y (mm) - table length direction", fontsize=10)
        ax.set_title(f"Landing point distribution heatmap (n={n})", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"\n保存しました: {out_path}")


if __name__ == "__main__":
    main()