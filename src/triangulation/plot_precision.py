"""
triangulate_points.py と同じ入力から、精度検証の結果をグラフとして可視化する
(Phase B-2の結果をゼミ発表用に見せるためのスクリプト)。

triangulate_points.pyの計算ロジックをそのまま使い、印字する代わりに
以下の2枚のグラフを1枚の画像として保存する:
    左: 卓球台を真上から見た図。実測位置(黒丸)と推定位置(色付き三角)を
        線で結んで、どちらの方向・どれくらいズレているかを見せる。
    右: 各点の誤差(mm)を棒グラフで表示し、RMS誤差を点線で重ねる。

使い方:
    triangulate_points.py と同じ引数で実行できる(--csv-out以外は同じ)。

    python plot_precision.py <カメラAフォルダ> <カメラBフォルダ> \
        <stereo_result.npz> <既知座標.csv> [--table-transform table_transform.npz] \
        [--ids id1,id2,...] [出力png]

例:
    python plot_precision.py captures\\known_a captures\\known_b \
        stereo_result.npz known_coords.csv --table-transform table_transform.npz \
        results\\20260918\\precision_plot.png
"""

import sys
import os
import re
import csv
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# triangulate.py がこのファイルと別フォルダにあっても見つけられるように、
# よくある置き場所(src\triangulation)も検索対象に加えておく。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [_THIS_DIR, os.path.join(_THIS_DIR, "src", "triangulation"),
                    os.path.join(_THIS_DIR, "..", "triangulation")]:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    from triangulate import load_calibration, build_projection_matrices, triangulate, apply_table_transform
except ModuleNotFoundError:
    print("[エラー] triangulate.py が見つかりません。plot_precision.py を "
          "triangulate.py と同じフォルダ(例: src\\triangulation\\)にコピーしてから実行してください。")
    sys.exit(1)

# ---- 日本語フォント設定(visualize_heatmap.pyと同じ方式) ----
_JP_FONT_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_jp_font = next((f for f in _JP_FONT_CANDIDATES if f in _available), None)
if _jp_font:
    matplotlib.rcParams["font.family"] = _jp_font
else:
    print("[警告] 日本語フォントが見つからなかったため、ラベルを英語表記にします。")
matplotlib.rcParams["axes.unicode_minus"] = False


def list_point_files(folder):
    result = {}
    for f in os.listdir(folder):
        if not f.endswith("_points.json"):
            continue
        point_id = f[: -len("_points.json")]
        path = os.path.join(folder, f)
        with open(path, "r", encoding="utf-8") as fp:
            pts = json.load(fp)
        if len(pts) == 0:
            continue
        result[point_id] = tuple(pts[0])
    return result


def load_true_coords(csv_path):
    result = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["id"]] = (float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"]))
    return result


def main():
    args = sys.argv[1:]

    table_transform_path = None
    if "--table-transform" in args:
        i = args.index("--table-transform")
        table_transform_path = args[i + 1]
        args = args[:i] + args[i + 2:]

    only_ids = None
    if "--ids" in args:
        i = args.index("--ids")
        only_ids = set(x.strip() for x in args[i + 1].split(","))
        args = args[:i] + args[i + 2:]

    out_path = None
    if len(args) == 5:
        out_path = args[4]
        args = args[:4]

    if len(args) != 4:
        print(__doc__)
        sys.exit(1)

    folder_a, folder_b, stereo_npz_path, csv_path = args
    if out_path is None:
        out_path = "precision_plot.png"

    pts_a = list_point_files(folder_a)
    pts_b = list_point_files(folder_b)
    true_coords = load_true_coords(csv_path)

    ids = sorted(set(pts_a) & set(pts_b) & set(true_coords))
    if only_ids is not None:
        ids = sorted(set(ids) & only_ids)

    print(f"対応がとれた点数: {len(ids)} / {len(true_coords)}")
    if not ids:
        print("[エラー] 対応する点が1つもありません。")
        sys.exit(1)

    K1, dist1, K2, dist2, R, T = load_calibration(stereo_npz_path)
    P1, P2 = build_projection_matrices(K1, K2, R, T)

    pts1 = np.array([pts_a[i] for i in ids], dtype=np.float64)
    pts2 = np.array([pts_b[i] for i in ids], dtype=np.float64)
    true_mm = np.array([true_coords[i] for i in ids], dtype=np.float64)

    estimated_mm = triangulate(P1, P2, K1, dist1, K2, dist2, pts1, pts2)

    if table_transform_path is not None:
        td = np.load(table_transform_path)
        estimated_mm = apply_table_transform(estimated_mm, td["R"], td["t"])
    else:
        print("[警告] --table-transformが指定されていないため、座標系が合っていない可能性があります。")

    diff = estimated_mm - true_mm
    dist_error = np.linalg.norm(diff, axis=1)
    xy_error = np.linalg.norm(diff[:, :2], axis=1)
    z_error = np.abs(diff[:, 2])
    rms = float(np.sqrt(np.mean(dist_error ** 2)))
    xy_rms = float(np.sqrt(np.mean(xy_error ** 2)))
    z_rms = float(np.sqrt(np.mean(z_error ** 2)))

    print(f"平均誤差: {np.mean(dist_error):.2f} mm / RMS誤差: {rms:.2f} mm")
    print(f"XY平面(着地位置)のみのRMS誤差: {xy_rms:.2f} mm")
    print(f"Z方向(高さ)のみのRMS誤差: {z_rms:.2f} mm")

    # ---- 描画 ----
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(20, 6.5))

    # 左: 真上から見た実測位置 vs 推定位置
    ax1.scatter(true_mm[:, 0], true_mm[:, 1], s=140, facecolor="white",
                edgecolor="black", linewidth=1.5, zorder=3,
                label="実測位置" if _jp_font else "True position")
    ax1.scatter(estimated_mm[:, 0], estimated_mm[:, 1], s=100, marker="^",
                facecolor="#E4572E", edgecolor="black", linewidth=1, zorder=4,
                label="推定位置" if _jp_font else "Estimated position")
    for i, pid in enumerate(ids):
        ax1.plot([true_mm[i, 0], estimated_mm[i, 0]], [true_mm[i, 1], estimated_mm[i, 1]],
                  color="#999999", linewidth=1, zorder=2)
        ax1.annotate(pid, (true_mm[i, 0], true_mm[i, 1]), textcoords="offset points",
                      xytext=(6, 8), fontsize=9)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_aspect("equal")
    ax1.set_title("実測位置と推定位置の比較(真上から)" if _jp_font else "True vs estimated position (top view)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    # 中央: 真横から見た図(X-Z)。Z方向(高さ)のズレをそのまま見せる。
    ax3.scatter(true_mm[:, 0], true_mm[:, 2], s=140, facecolor="white",
                edgecolor="black", linewidth=1.5, zorder=3,
                label="実測位置" if _jp_font else "True position")
    ax3.scatter(estimated_mm[:, 0], estimated_mm[:, 2], s=100, marker="^",
                facecolor="#E8A33D", edgecolor="black", linewidth=1, zorder=4,
                label="推定位置" if _jp_font else "Estimated position")
    for i in range(len(ids)):
        ax3.plot([true_mm[i, 0], estimated_mm[i, 0]], [true_mm[i, 2], estimated_mm[i, 2]],
                  color="#E8A33D", linewidth=1.5, zorder=2)
        ax3.annotate(ids[i], (true_mm[i, 0], true_mm[i, 2]), textcoords="offset points",
                      xytext=(4, 8), fontsize=8)
    ax3.axhline(0, color="black", linewidth=1.5, zorder=1,
                label="卓球台面(Z=0)" if _jp_font else "Table surface (Z=0)")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)  ※卓球台からの高さ" if _jp_font else "Z (mm) - height above table")
    ax3.set_title("Z方向のズレ(真横から)" if _jp_font else "Z-axis deviation (side view)")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, linewidth=0.4, alpha=0.5)

    # 右: 点ごとの誤差(mm)棒グラフ + RMS線
    # id内の数字(known_01のような番号)の小さい順に並べる。数字が無ければ文字列順。
    def _id_sort_key(pid):
        m = re.search(r"\d+", pid)
        return (int(m.group()) if m else float("inf"), pid)

    order = sorted(range(len(ids)), key=lambda i: _id_sort_key(ids[i]))
    bar_w = 0.35
    xpos = np.arange(len(ids))
    ax2.bar(xpos - bar_w / 2, xy_error[order], width=bar_w, color="#4C8FC0",
            label="XY平面の誤差(着地位置)" if _jp_font else "XY-plane error (landing position)")
    ax2.bar(xpos + bar_w / 2, z_error[order], width=bar_w, color="#E8A33D",
            label="Z方向の誤差(高さ)" if _jp_font else "Z-axis error (height)")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([ids[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax2.axhline(xy_rms, color="#4C8FC0", linestyle="--", linewidth=1.5,
                label=f"RMS(XY) {xy_rms:.1f}mm" if _jp_font else f"RMS (XY) {xy_rms:.1f}mm")
    ax2.axhline(z_rms, color="#E8A33D", linestyle="--", linewidth=1.5,
                label=f"RMS(Z) {z_rms:.1f}mm" if _jp_font else f"RMS (Z) {z_rms:.1f}mm")
    ax2.set_ylabel("誤差 (mm)" if _jp_font else "Error (mm)")
    ax2.set_title("点ごとの誤差(XY平面 / Z方向)" if _jp_font else "Per-point error (XY / Z)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle(f"精度検証結果 (n={len(ids)}, XY:RMS={xy_rms:.1f}mm / Z:RMS={z_rms:.1f}mm)" if _jp_font else
                  f"Precision check (n={len(ids)}, XY:RMS={xy_rms:.1f}mm / Z:RMS={z_rms:.1f}mm)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"\n保存しました: {out_path}")


if __name__ == "__main__":
    main()