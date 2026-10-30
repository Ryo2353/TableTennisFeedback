# -*- coding: utf-8 -*-
r"""
track_balls.py の出力(軌道ごとの3D位置)から、1球ごとの着地点を求める。

着地点の求め方(今の fit_bounce.py と同じ「3点内挿」。見直しは後回しにしている):
    1. 軌道の中で、Z が前後の点より低く、かつ --z-bounce(標準 50mm)以下の点を「弾んだ所」とみなす。
    2. その点と前後1点ずつの3点で、時間に対する Z の2次関数を決め、一番低くなる時刻を着地の瞬間とする
       (谷の形にならない・3点の範囲から外れるときは、その点の時刻をそのまま使う)。
    3. X, Y は同じ3点の直線(時間の1次式)で、その瞬間の位置を求める。
    4. 計測範囲(X 0〜1165、Y 0〜840)の中で弾んだ最初の所を、その球の着地点とする。
       範囲の外で弾んだだけの球、弾んだ所がない球は「着地点なし」として画面に出す。

fit_bounce.py との違い:
    fit_bounce.py は「コマが10以上空いたら別の球」で区切っていたので、2球が同時に飛ぶと区切れなかった。
    ここでは track_balls.py が分けた軌道の番号(track)ごとに求める。

使い方:
    python src\triangulation\landings_from_tracks.py <track_balls.py の出力csv> [<出力csv>]
例:
    python src\triangulation\landings_from_tracks.py results\20261023\rec02_tracks.csv
    → results\20261023\rec02_tracks_landings.csv(x_mm, y_mm の列があるので project_heatmap.py にそのまま渡せる)
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

FPS = 227.0
TABLE_X = (0.0, 1165.0)
TABLE_Y = (0.0, 840.0)


def bounce_at(ts, X, i):
    """i 番目の点(Zが低い点)とその前後で、着地の時刻と位置を求める"""
    lo, hi = max(0, i - 1), min(len(ts) - 1, i + 1)
    if hi - lo < 2:
        return ts[i], X[i]
    t3, x3 = ts[lo:hi + 1], X[lo:hi + 1]
    a, b, _ = np.polyfit(t3, x3[:, 2], 2)
    t_star = ts[i]
    if a > 0:
        tv = -b / (2 * a)
        if t3[0] <= tv <= t3[-1]:
            t_star = tv
    pos = np.array([np.polyval(np.polyfit(t3, x3[:, k], 1), t_star) for k in range(2)]
                   + [np.polyval(np.polyfit(t3, x3[:, 2], 2), t_star)])
    return t_star, pos


def in_table(p):
    return TABLE_X[0] <= p[0] <= TABLE_X[1] and TABLE_Y[0] <= p[1] <= TABLE_Y[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks_csv")
    ap.add_argument("out_csv", nargs="?", default=None)
    ap.add_argument("--z-bounce", type=float, default=50.0, help="Z がこれ(mm)以下の谷を「弾んだ所」とみなす")
    a = ap.parse_args()
    out = a.out_csv or os.path.splitext(a.tracks_csv)[0] + "_landings.csv"

    rows = defaultdict(list)
    with open(a.tracks_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[int(r["track"])].append((int(r["delta"]), float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])))

    results, skipped = [], []
    for tr in sorted(rows):
        pts = sorted(rows[tr])
        ts = np.array([p[0] for p in pts], float) / FPS
        X = np.array([p[1:] for p in pts], float)
        z = X[:, 2]
        mins = [i for i in range(len(z))
                if z[i] <= a.z_bounce
                and (i == 0 or z[i] <= z[i - 1]) and (i == len(z) - 1 or z[i] <= z[i + 1])]
        found = None
        outside = []
        for i in mins:
            t_star, p = bounce_at(ts, X, i)
            if in_table(p):
                found = (t_star, p, z[i])
                break
            outside.append(p)
        if found:
            results.append((tr, found[0], found[1], found[2]))
        elif outside:
            p = outside[0]
            skipped.append((tr, f"計測範囲の外で弾んだ(X {p[0]:.0f}, Y {p[1]:.0f})"))
        else:
            skipped.append((tr, f"弾んだ所がない(一番低いZ {z.min():.0f}mm)"))

    print(f"軌道 {len(rows)} 本 → 着地点 {len(results)} 個")
    print("  軌道  時刻(秒)   X(mm)   Y(mm)  弾んだ所のZ(mm)")
    for tr, t, p, zm in results:
        print(f"  {tr:>4}  {t:8.2f}  {p[0]:6.0f}  {p[1]:6.0f}  {zm:8.0f}")
    for tr, why in skipped:
        print(f"  {tr:>4}  着地点なし: {why}")

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "t_s", "x_mm", "y_mm", "z_min_mm"])
        for tr, t, p, zm in results:
            w.writerow([tr, f"{t:.4f}", f"{p[0]:.1f}", f"{p[1]:.1f}", f"{zm:.1f}"])
    print(f"保存: {out}")


if __name__ == "__main__":
    main()