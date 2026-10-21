"""
既知位置(実座標が分かっている点)の画像上座標から三角測量を行い、
実測値との誤差(mm)を計算する — Phase B-2(精度検証)用スクリプト。

前提となるファイル構成:
    1. stereo_calibrate_charuco.py で作成した stereo_result.npz
    2. pick_points.py で拾った、各既知位置のカメラA・カメラB画像上の座標
       (例: known_01_points.json, known_02_points.json, ... がカメラA用
        フォルダとカメラB用フォルダそれぞれに1つずつあり、各ファイルの
        中身は点1個 [[u, v]] のみであること)
    3. 各既知位置の実世界座標をまとめたCSVファイル(自分で作成する)
       形式:
           id,x_mm,y_mm,z_mm
           known_01,100,200,0
           known_02,600,200,0
           ...
       id列は、pick_pointsで作った "<id>_points.json" の <id> 部分と
       一致させること。x,y,zは卓球台の原点(給球機側エンドラインの左角、
       X+=右, Y+=ネット方向, Z+=卓球台面から上)基準、単位mm。
    4. (--table-transform を使う場合) compute_table_pose.py で作成した
       table_transform.npz。triangulate()の結果はカメラA基準の座標なので、
       卓球台基準の実座標とそのまま比較すると誤差が正しく出ない。
       --table-transform を指定すると、比較前にカメラA基準->卓球台基準の
       変換を適用してから誤差を計算する。

使い方:
    python triangulate_points.py <カメラAフォルダ> <カメラBフォルダ> \
        <stereo_result.npz> <既知座標.csv> [--table-transform table_transform.npz] \
        [--ids id1,id2,...]

    --ids を指定すると、そのidの点だけで誤差を集計する。
    compute_table_pose.py の --ids で姿勢推定に使った点を除いた
    holdout(検証専用)の点だけを見たい場合などに使う。

例:
    python triangulate_points.py captures\\known_a captures\\known_b \
        stereo_result.npz known_coords.csv --table-transform table_transform.npz \
        --ids known03,known07,known08,known10

出力:
    各点ごとの推定座標・実座標・誤差(mm)、および全体のRMS誤差。
"""

import sys
import os
import csv
import json
import numpy as np

from triangulate import load_calibration, build_projection_matrices, triangulate, apply_table_transform


def list_point_files(folder):
    """{id: (u, v)} の辞書を返す。<id>_points.json というファイルのみ対象。"""
    result = {}
    for f in os.listdir(folder):
        if not f.endswith("_points.json"):
            continue
        point_id = f[: -len("_points.json")]
        path = os.path.join(folder, f)
        with open(path, "r", encoding="utf-8") as fp:
            pts = json.load(fp)
        if len(pts) != 1:
            print(f"  [警告] {f} は点が{len(pts)}個あります(1個を想定)。先頭の点のみ使用します。")
        if len(pts) == 0:
            continue
        result[point_id] = tuple(pts[0])
    return result


def load_true_coords(csv_path):
    """{id: (x_mm, y_mm, z_mm)} の辞書を返す。"""
    result = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["id"]] = (
                float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])
            )
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

    if len(args) != 4:
        print(__doc__)
        sys.exit(1)

    folder_a, folder_b, stereo_npz_path, csv_path = args

    pts_a = list_point_files(folder_a)
    pts_b = list_point_files(folder_b)
    true_coords = load_true_coords(csv_path)

    ids = sorted(set(pts_a) & set(pts_b) & set(true_coords))
    if only_ids is not None:
        unknown = only_ids - set(ids)
        if unknown:
            print(f"[警告] --ids で指定されたが対応が取れないid: {sorted(unknown)}")
        ids = sorted(set(ids) & only_ids)
    missing_a = set(true_coords) - set(pts_a)
    missing_b = set(true_coords) - set(pts_b)
    if missing_a:
        print(f"[警告] カメラA側で見つからないid: {sorted(missing_a)}")
    if missing_b:
        print(f"[警告] カメラB側で見つからないid: {sorted(missing_b)}")

    print(f"\n対応がとれた点数: {len(ids)} / {len(true_coords)}")
    if not ids:
        print("[エラー] 対応する点が1つもありません。ファイル名(id)とCSVのid列が一致しているか確認してください。")
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
        print(f"\n[情報] {table_transform_path} の変換(カメラA基準->卓球台基準)を適用しました。")
    else:
        print("\n[警告] --table-transform が指定されていません。"
              "triangulate()の結果はカメラA基準の座標なので、卓球台基準の実座標とは"
              "そのままでは比較できません(座標系が異なります)。compute_table_pose.py で"
              "変換を求めてから --table-transform で指定してください。")

    diff = estimated_mm - true_mm
    dist_error = np.linalg.norm(diff, axis=1)

    print("\n=== 点ごとの結果 ===")
    print(f"{'id':<12}{'推定(X,Y,Z)mm':<32}{'実測(X,Y,Z)mm':<28}{'誤差mm':<10}")
    for i, pid in enumerate(ids):
        est = estimated_mm[i]
        tru = true_mm[i]
        print(
            f"{pid:<12}"
            f"({est[0]:7.1f},{est[1]:7.1f},{est[2]:7.1f})   "
            f"({tru[0]:7.1f},{tru[1]:7.1f},{tru[2]:7.1f})   "
            f"{dist_error[i]:7.2f}"
        )

    rms = float(np.sqrt(np.mean(dist_error ** 2)))
    print(f"\n平均誤差: {np.mean(dist_error):.2f} mm")
    print(f"最大誤差: {np.max(dist_error):.2f} mm ({ids[int(np.argmax(dist_error))]})")
    print(f"RMS誤差: {rms:.2f} mm")

    # 軸ごとの誤差も見ておく(どの軸に偏っているか分かると原因調査しやすい)
    axis_rms = np.sqrt(np.mean(diff ** 2, axis=0))
    print(f"\n軸ごとのRMS誤差: X={axis_rms[0]:.2f}mm  Y={axis_rms[1]:.2f}mm  Z={axis_rms[2]:.2f}mm")


if __name__ == "__main__":
    main()