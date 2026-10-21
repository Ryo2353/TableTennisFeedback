"""
triangulate_trajectory.py が出力した軌跡CSV(delta, ts_a, ts_b, x_mm, y_mm, z_mm)から、
ボールが卓球台面に着地した地点(X, Y)を推定する。

考え方:
    「観測できたフレームの中でZが一番小さかったフレーム」をそのまま着地点と
    するのが一番シンプルだが、そのフレームがちょうど着地の瞬間とは限らない
    (前後のフレームの間のどこかで本当は着地している)ため、それをそのまま
    使うと真の着地点とズレる。

    かといって、バウンド前の点だけを多数使って2次関数をフィットし、頂点を
    遠くまで外挿する方式は、わずかなノイズで不安定になることが検証で分かった
    (谷の丸まりがまだ見えていない、ほぼ直線的なデータから頂点の位置を
    言い当てようとすると、2次の係数の推定が不安定になるため)。

    そこで、「Zが最小だった点と、その前後1点ずつ、計3点」だけを使って
    2次関数を当てはめる。3点あれば2次関数(a,b,c)はただ1通りに決まる
    (連立方程式を解くだけ)ので、最小二乗フィットのような不安定さが出ない。
    また、求めた頂点は観測点の外側への外挿ではなく、観測点の間を埋める
    内挿になるので、精度・安定性の両面で望ましい。
    (前後の点の片方はバウンド後の点である可能性があるが、隣接1点だけなら
    影響は小さく、実用上は許容できると判断している。)

使い方:
    python fit_bounce.py <軌跡csv>

例:
    python fit_bounce.py results\\20260918\\ball_detection\\naka_oku_trajectory.csv
"""

import sys
import csv
import numpy as np

# Zがこの値以下であれば「ほぼ卓球台に触れている」とみなす(mm)。
# 現在の三角測量精度(±20mm程度)を踏まえた目安。
Z_BOUNCE_THRESHOLD_MM = 50.0


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]

    all_deltas, all_xs, all_ys, all_zs = [], [], [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_deltas.append(float(row["delta"]))
            all_xs.append(float(row["x_mm"]))
            all_ys.append(float(row["y_mm"]))
            all_zs.append(float(row["z_mm"]))

    order = np.argsort(all_deltas)
    all_deltas = np.array(all_deltas)[order]
    all_xs = np.array(all_xs)[order]
    all_ys = np.array(all_ys)[order]
    all_zs = np.array(all_zs)[order]

    print(f"総点数: {len(all_deltas)}")

    # ---- deltaが大きく飛んでいるところで「別の球」として分割 ----
    GAP_THRESHOLD = 10
    gaps = np.where(np.diff(all_deltas) > GAP_THRESHOLD)[0]
    segment_starts = [0] + (gaps + 1).tolist()
    segment_ends = gaps.tolist() + [len(all_deltas) - 1]

    print(f"検出された軌跡の塊: {len(segment_starts)}個\n")

    valid_landings = []

    for seg_i, (s, e) in enumerate(zip(segment_starts, segment_ends)):
        deltas = all_deltas[s:e + 1]
        xs = all_xs[s:e + 1]
        ys = all_ys[s:e + 1]
        zs = all_zs[s:e + 1]

        print(f"--- 軌跡{seg_i + 1}: delta={deltas[0]:.0f}〜{deltas[-1]:.0f} ({len(deltas)}点) ---")

        min_idx = int(np.argmin(zs))
        raw_delta = deltas[min_idx]
        raw_z = zs[min_idx]
        print(f"  Z最小値(生データ): {raw_z:.1f}mm (delta={raw_delta:.0f})")

        if 0 < min_idx < len(deltas) - 1:
            # ---- 前後1点ずつ、計3点で2次関数を一意に決めて内挿する ----
            t3 = deltas[min_idx - 1:min_idx + 2]
            z3 = zs[min_idx - 1:min_idx + 2]
            x3 = xs[min_idx - 1:min_idx + 2]
            y3 = ys[min_idx - 1:min_idx + 2]

            coef_z = np.polyfit(t3, z3, 2)  # 3点を通る2次関数はただ1つに決まる
            a, b, c = coef_z

            if a > 0 and (t3.min() <= -b / (2 * a) <= t3.max()):
                bounce_delta = -b / (2 * a)
                bounce_z = np.polyval(coef_z, bounce_delta)
                # X, Yは同じ3点を使った1次(線形)内挿で求める
                coef_x = np.polyfit(t3, x3, 1)
                coef_y = np.polyfit(t3, y3, 1)
                bounce_x = np.polyval(coef_x, bounce_delta)
                bounce_y = np.polyval(coef_y, bounce_delta)
                print(f"  → 前後1点を使った内挿: delta={bounce_delta:.1f}, "
                      f"Z={bounce_z:.1f}mm, X={bounce_x:.1f}mm, Y={bounce_y:.1f}mm")
            else:
                print("  [警告] 3点での内挿がうまくいかなかったため、生データの最小値フレームを使います。")
                bounce_delta, bounce_z = raw_delta, raw_z
                bounce_x, bounce_y = xs[min_idx], ys[min_idx]
        else:
            print("  [警告] Z最小値がデータの端(前後どちらかに点がない)なので、生データの最小値フレームを使います。")
            bounce_delta, bounce_z = raw_delta, raw_z
            bounce_x, bounce_y = xs[min_idx], ys[min_idx]

        print(f"  着地点(推定): X = {bounce_x:.1f} mm, Y = {bounce_y:.1f} mm")

        in_table = (0 <= bounce_x <= 1165) and (0 <= bounce_y <= 840)
        close_enough = bounce_z <= Z_BOUNCE_THRESHOLD_MM

        if in_table and close_enough:
            print("  → 卓球台範囲内、かつ卓球台面に十分近い。有効な着地点として採用します。\n")
            valid_landings.append((seg_i + 1, bounce_x, bounce_y, bounce_z))
        elif not close_enough:
            print(f"  → Z最小値が{Z_BOUNCE_THRESHOLD_MM}mmより大きく、卓球台面まで十分近づいていません。"
                  "着地の瞬間を捉えられていない(検出が途切れた)可能性があります。着地点としては採用しません。\n")
        else:
            print("  → 卓球台範囲外。台の上空は通過しているが着地はしていない"
                  "(打球が台を飛び越えた等)可能性があります。着地点としては採用しません。\n")

    print("=== 有効と判定された着地点 ===")
    if valid_landings:
        for seg_i, x, y, z in valid_landings:
            print(f"  軌跡{seg_i}: X = {x:.1f} mm, Y = {y:.1f} mm (Z最小値 {z:.1f}mm)")
    else:
        print("  (なし)")

    # ---- 有効な着地点をCSVに保存(run_all.pyでの自動集約用) ----
    landings_csv = csv_path.rsplit(".csv", 1)[0] + "_landings.csv"
    with open(landings_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "x_mm", "y_mm", "z_min_mm"])
        for seg_i, x, y, z in valid_landings:
            writer.writerow([seg_i, f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"])
    print(f"\n有効な着地点を保存しました: {landings_csv}")


if __name__ == "__main__":
    main()