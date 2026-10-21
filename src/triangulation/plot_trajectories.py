# -*- coding: utf-8 -*-
"""
検出できたボールの軌道を一覧にする(表と図)。

diagnose_triangulation.py が録画フォルダに出す diagnose_points.csv を読み、
delta が 10 以上飛んだところで区切って「1本の軌道」とする(fit_bounce.py と同じ区切り方)。

出力:
  1. 画面に、軌道ごとの一覧表(コマ数、2台でつじつまの合う点の数、X・Y・Z の範囲)
  2. <録画フォルダ>/trajectories_overview.png
       左: 台を真上から見た図(X-Y)、右: 高さ Z の時間変化
       ● = 2台でつじつまの合う点(再投影誤差 5画素以下)
       × = つじつまの合わない点(2台が別のボールを見ていた可能性が高い)
  3. <録画フォルダ>/trajectories_each.png
       軌道ごとの Z の時間変化(点が5個以上の軌道だけ。並べて見比べる用)

使い方:
    python plot_trajectories.py <録画フォルダ>
例:
    python plot_trajectories.py data\\captures\\20261016_190114\\rec02
"""

import csv
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

_JP = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"]
_avail = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_font = next((f for f in _JP if f in _avail), None)
if _font:
    matplotlib.rcParams["font.family"] = _font
matplotlib.rcParams["axes.unicode_minus"] = False

TABLE_W, TABLE_L = 1165.0, 840.0
GAP = 10          # delta がこれ以上飛んだら別の軌道
GOOD_PX = 5.0     # 再投影誤差がこれ以下なら「つじつまが合う」
C_GOOD = "#2a6fdb"   # 青
C_BAD = "#d1495b"    # 赤
INK = "#333333"
MUTED = "#999999"


def load(rec_dir):
    path = os.path.join(rec_dir, "diagnose_points.csv")
    if not os.path.exists(path):
        print(f"[エラー] {path} がありません。先に diagnose_triangulation.py を実行してください。")
        sys.exit(1)
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["delta"]), max(float(r["err_a_px"]), float(r["err_b_px"])),
                         float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])))
    rows.sort()
    a = np.array(rows, dtype=float)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def split(delta):
    segs, start = [], 0
    for i in range(1, len(delta) + 1):
        if i == len(delta) or delta[i] - delta[i - 1] >= GAP:
            segs.append(np.arange(start, i))
            start = i
    return segs


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    rec = sys.argv[1]
    delta, err, x, y, z = load(rec)
    good = err <= GOOD_PX
    # 再投影誤差がとても大きい点は、3D位置が意味をなさないので図には描かない(表には数える)
    sane = err < 200
    segs = split(delta)

    # ---- 1. 一覧表 ----
    print(f"点の総数 {len(delta)}  軌道の数 {len(segs)}  つじつまの合う点 {int(good.sum())}\n")
    print(f"{'#':>3} {'delta':>13} {'点':>4} {'合う点':>5}   {'X(mm)':>13} {'Y(mm)':>13} {'Z(mm)':>13}   メモ")
    for k, s in enumerate(segs, 1):
        g = s[good[s]]
        use = g if len(g) else s
        note = []
        if len(g) == 0:
            note.append("合う点なし")
        if len(g) and (z[g] < -50).any():
            note.append("合う点で台より下あり")
        if len(g) and ((x[g] < 0) | (x[g] > TABLE_W) | (y[g] < 0) | (y[g] > TABLE_L)).any():
            note.append("台の範囲外あり")
        print(f"{k:>3} {delta[s[0]]:>6}〜{delta[s[-1]]:<6} {len(s):>4} {len(g):>5}   "
              f"{x[use].min():>5.0f}〜{x[use].max():<6.0f} {y[use].min():>5.0f}〜{y[use].max():<6.0f} "
              f"{z[use].min():>5.0f}〜{z[use].max():<6.0f}  {'、'.join(note)}")

    # ---- 2. 全体図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.3]})
    ax1.add_patch(Rectangle((0, 0), TABLE_W, TABLE_L, fill=False, edgecolor=INK, linewidth=1.5))
    ax1.axvline(TABLE_W / 2, color=MUTED, linewidth=1, linestyle="--")
    for s in segs:
        ss = s[sane[s]]
        if len(ss) > 1:
            ax1.plot(x[ss], y[ss], color=MUTED, linewidth=0.8, zorder=1)
    m = sane & ~good
    ax1.scatter(x[m], y[m], marker="x", s=22, color=C_BAD, linewidths=1.2, zorder=2,
                label=f"つじつまが合わない ({int((~good).sum())})")
    ax1.scatter(x[good], y[good], s=26, color=C_GOOD, edgecolor="white", linewidth=0.6, zorder=3,
                label=f"2台でつじつまが合う ({int(good.sum())})")
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (mm)  幅方向")
    ax1.set_ylabel("Y (mm)  長さ方向")
    ax1.set_title("真上から見た位置(枠 = 計測範囲)")
    ax1.legend(loc="upper right", fontsize=9, frameon=False)
    ax1.grid(color="#e6e6e6", linewidth=0.6)

    t = delta / 227.0
    ax2.axhline(0, color=INK, linewidth=1.2)
    ax2.axhspan(-50, 50, color="#eeeeee", zorder=0)
    for s in segs:
        ss = s[sane[s]]
        if len(ss) > 1:
            ax2.plot(t[ss], z[ss], color=MUTED, linewidth=0.8, zorder=1)
    ax2.scatter(t[m], z[m], marker="x", s=22, color=C_BAD, linewidths=1.2, zorder=2)
    ax2.scatter(t[good], z[good], s=26, color=C_GOOD, edgecolor="white", linewidth=0.6, zorder=3)
    ax2.set_xlabel("録画開始からの時間 (秒)")
    ax2.set_ylabel("Z (mm)  台の面からの高さ")
    ax2.set_title("高さの変化(灰色の帯 = 台面 ±50mm)")
    ax2.grid(color="#e6e6e6", linewidth=0.6)
    fig.tight_layout()
    out1 = os.path.join(rec, "trajectories_overview.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # ---- 3. 軌道ごとの小さな図 ----
    big = [(k, s) for k, s in enumerate(segs, 1) if len(s) >= 5]
    if big:
        cols = 4
        rows_n = int(np.ceil(len(big) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, 2.8 * rows_n), squeeze=False)
        for ax, (k, s) in zip(axes.ravel(), big):
            g = good[s]
            ss = s[sane[s]]
            ax.axhline(0, color=INK, linewidth=1)
            ax.axhspan(-50, 50, color="#eeeeee", zorder=0)
            ax.plot(delta[ss], z[ss], color=MUTED, linewidth=0.8)
            ax.scatter(delta[s][~g & sane[s]], z[s][~g & sane[s]], marker="x", s=18, color=C_BAD)
            ax.scatter(delta[s][g], z[s][g], s=20, color=C_GOOD, edgecolor="white", linewidth=0.5)
            ax.set_title(f"軌道{k}  delta {delta[s[0]]}〜{delta[s[-1]]}  合う点 {int(g.sum())}/{len(s)}",
                         fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(color="#e6e6e6", linewidth=0.6)
        for ax in axes.ravel()[len(big):]:
            ax.axis("off")
        fig.supylabel("Z (mm)")
        fig.supxlabel("delta(コマ)")
        fig.tight_layout()
        out2 = os.path.join(rec, "trajectories_each.png")
        fig.savefig(out2, dpi=130)
        plt.close(fig)
        print(f"\n保存しました: {out1}\n            {out2}")
    else:
        print(f"\n保存しました: {out1}")


if __name__ == "__main__":
    main()