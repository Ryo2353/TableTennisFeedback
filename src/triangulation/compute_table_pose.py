"""
solvePnPで「カメラA基準の座標系」→「卓球台基準の座標系」への変換を求める。

三角測量(triangulate.py)が返す3D座標はカメラA基準(カメラAの光学中心が原点)。
これを卓球台基準(マシン側エンドライン左角が原点)の座標に変換するために、
卓球台上の実座標が分かっている点とカメラA画像上のピクセル座標の対応から、
カメラAの卓球台に対する位置・姿勢(R, t)をsolvePnPで求める。

点の与え方は2種類あり、どちらか一方でも、両方組み合わせてもよい:

  (A) 4隅(--corners)
      pick_points.pyで、卓球台の4隅を以下の順序でクリックし、
      1つの *_points.json(4点)として保存したもの。
          1. エンドライン左(原点)
          2. エンドライン右
          3. プリント切れ目・左
          4. プリント切れ目・右
      4隅の実座標(mm)は src/table_config.py の設定から作る(--table で台を指定できる)。
      新しい台では 3, 4 はネットの真下の左右の端になる。
      (全点Z=0の平面上なので、奥行き方向の姿勢誤差に弱い)

  (B) 既知点フォルダ + CSV(--known-folder / --known-csv)
      Phase B-2の精度検証で使った <id>_points.json 群(カメラA側)と
      known_coords.csv(id,x_mm,y_mm,z_mm)をそのまま流用する。
      Z=0でない点(ボールを持ち上げて置いた点など)が含まれていれば、
      4隅だけでは弱い奥行き方向の姿勢もより正確に求まりやすい。

      --ids で使う点を絞り込める(例: --ids known01,known02,known04)。
      精度検証(triangulate_points.py)と同じ点を姿勢推定にも使ってしまうと
      「自分で合わせたデータで自分を採点する」ことになり評価が甘くなるため、
      姿勢推定に使う点(calibration用)と検証に使う点(validation用)は
      分けることを推奨する。

  (C) 追加の任意点(--extra-folder / --extra-csv)
      (B)と同じ「フォルダ + CSV」形式だが、精度検証用のknown01〜のセットとは
      完全に別の点(卓球台のセンターラインの交点など)を渡すための枠。
      検証用の既知点を一切使わずに姿勢推定の点数を増やせるので、
      (B)のように検証点を分割する必要がなくなる(全既知点をそのまま
      holdoutとして検証に使える)。

(A)(B)両方指定した場合は、点を全て合わせてsolvePnPを解く(4隅だけより
点数が増え、姿勢の安定性が上がることが期待できる)。

使い方:
    python compute_table_pose.py --calib <calibration_result.npz> --out <出力先.npz> \
        [--corners <4隅points.json>] \
        [--known-folder <カメラAの既知点フォルダ> --known-csv <known_coords.csv>]

例(4隅のみ、従来通り):
    python compute_table_pose.py --calib results/20260918/camera_a/calibration_result.npz \
        --out results/20260918/table_transform.npz \
        --corners results/20260918/precision_check/corners/table_corners_points.json

例(既知点10個のみ):
    python compute_table_pose.py --calib results/20260918/camera_a/calibration_result.npz \
        --out results/20260918/table_transform_v2.npz \
        --known-folder results/20260918/precision_check/camera_a \
        --known-csv results/20260918/precision_check/known_coords.csv

例(4隅+既知点10個、計14点):
    python compute_table_pose.py --calib results/20260918/camera_a/calibration_result.npz \
        --out results/20260918/table_transform_v2.npz \
        --corners results/20260918/precision_check/corners/table_corners_points.json \
        --known-folder results/20260918/precision_check/camera_a \
        --known-csv results/20260918/precision_check/known_coords.csv

例(4隅+センターライン2点、計6点。既知点10個は一切使わないので
   triangulate_points.pyで全10点をそのまま検証に使える):
    python compute_table_pose.py --calib results/20260918/camera_a/calibration_result.npz \
        --out results/20260918/table_transform_v3.npz \
        --corners results/20260918/precision_check/corners/table_corners_points.json \
        --extra-folder results/20260918/precision_check/corners \
        --extra-csv results/20260918/precision_check/corners/extra_coords.csv

出力(指定した出力先.npz)には R, t が保存され、triangulate_points.py の
--table-transform で使う。
"""

import sys
import os
import csv
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from table_config import get_table

def table_corners(table_name):
    """卓球台の4隅の実座標(mm)と名前。原点=マシン側エンドライン左角。大きさは table_config.py"""
    w, l, far = get_table(table_name)
    corners = np.array([
        [0.0, 0.0, 0.0],   # 1. エンドライン左(原点)
        [w, 0.0, 0.0],     # 2. エンドライン右
        [0.0, l, 0.0],     # 3. 遠い側の辺・左
        [w, l, 0.0],       # 4. 遠い側の辺・右
    ], dtype=np.float64)
    return corners, ["エンドライン左", "エンドライン右", f"{far}・左", f"{far}・右"]


def pop_option(args, flag):
    if flag in args:
        i = args.index(flag)
        value = args[i + 1]
        return value, args[:i] + args[i + 2:]
    return None, args


def load_known_points(folder, csv_path, only_ids=None):
    """既知点フォルダ + CSVから、(obj_points(N,3), img_points(N,2), labels) を返す。

    only_ids を指定すると、そのidのみに絞り込む
    (キャリブレーション用と検証用で点を分けたい場合に使う)。
    """
    pts_img = {}
    for f in os.listdir(folder):
        if not f.endswith("_points.json"):
            continue
        point_id = f[: -len("_points.json")]
        with open(os.path.join(folder, f), "r", encoding="utf-8") as fp:
            pts = json.load(fp)
        if len(pts) != 1:
            print(f"  [警告] {f} は点が{len(pts)}個あります(1個を想定)。先頭の点のみ使用します。")
        if len(pts) == 0:
            continue
        pts_img[point_id] = tuple(pts[0])

    true_coords = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_coords[row["id"]] = (
                float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])
            )

    ids = sorted(set(pts_img) & set(true_coords))
    if only_ids is not None:
        unknown = set(only_ids) - set(ids)
        if unknown:
            print(f"  [警告] --ids で指定されたが見つからないid: {sorted(unknown)}")
        ids = sorted(set(ids) & set(only_ids))
    missing = set(true_coords) - set(pts_img)
    if missing and only_ids is None:
        print(f"  [警告] {folder} で見つからないid: {sorted(missing)}")

    obj = np.array([true_coords[i] for i in ids], dtype=np.float64)
    img = np.array([pts_img[i] for i in ids], dtype=np.float64)
    return obj, img, ids


def main():
    args = sys.argv[1:]

    calib_path, args = pop_option(args, "--calib")
    out_path, args = pop_option(args, "--out")
    corners_json_path, args = pop_option(args, "--corners")
    known_folder, args = pop_option(args, "--known-folder")
    known_csv, args = pop_option(args, "--known-csv")
    ids_str, args = pop_option(args, "--ids")
    only_ids = set(x.strip() for x in ids_str.split(",")) if ids_str else None
    extra_folder, args = pop_option(args, "--extra-folder")
    extra_csv, args = pop_option(args, "--extra-csv")
    table_name, args = pop_option(args, "--table")
    TABLE_CORNERS_MM, CORNER_LABELS = table_corners(table_name)

    if calib_path is None or out_path is None:
        print(__doc__)
        sys.exit(1)
    if corners_json_path is None and known_folder is None and extra_folder is None:
        print("[エラー] --corners / --known-folder+--known-csv / --extra-folder+--extra-csv の"
              "少なくとも1つを指定してください。")
        print(__doc__)
        sys.exit(1)
    if (known_folder is None) != (known_csv is None):
        print("[エラー] --known-folder と --known-csv は両方セットで指定してください。")
        sys.exit(1)
    if (extra_folder is None) != (extra_csv is None):
        print("[エラー] --extra-folder と --extra-csv は両方セットで指定してください。")
        sys.exit(1)

    obj_points_list = []
    img_points_list = []
    labels = []

    if corners_json_path is not None:
        with open(corners_json_path, "r", encoding="utf-8") as f:
            pts = json.load(f)
        if len(pts) != len(TABLE_CORNERS_MM):
            print(f"[エラー] 4隅は{len(TABLE_CORNERS_MM)}点である必要がありますが、{len(pts)}点でした。"
                  f" pick_points.pyでのクリック順序も、このファイルのTABLE_CORNERS_MMの順序と揃っているか確認してください。")
            sys.exit(1)
        obj_points_list.append(TABLE_CORNERS_MM)
        img_points_list.append(np.array(pts, dtype=np.float64))
        labels.extend(CORNER_LABELS)
        print(f"[情報] 4隅を追加: {corners_json_path}")

    if known_folder is not None:
        obj_k, img_k, ids_k = load_known_points(known_folder, known_csv, only_ids=only_ids)
        if len(ids_k) == 0:
            print(f"[エラー] {known_folder} / {known_csv} から対応点が1つも取れませんでした。")
            sys.exit(1)
        obj_points_list.append(obj_k)
        img_points_list.append(img_k)
        labels.extend(ids_k)
        print(f"[情報] 既知点を追加: {known_folder} ({len(ids_k)}点, {sorted(ids_k)})")

    if extra_folder is not None:
        obj_e, img_e, ids_e = load_known_points(extra_folder, extra_csv)
        if len(ids_e) == 0:
            print(f"[エラー] {extra_folder} / {extra_csv} から対応点が1つも取れませんでした。")
            sys.exit(1)
        obj_points_list.append(obj_e)
        img_points_list.append(img_e)
        labels.extend(ids_e)
        print(f"[情報] 追加点を追加: {extra_folder} ({len(ids_e)}点, {sorted(ids_e)})")

    object_points = np.vstack(obj_points_list)
    img_points = np.vstack(img_points_list)
    print(f"\n合計 {len(object_points)} 点でPnPを解きます。")

    d = np.load(calib_path)
    K, dist = d["camera_matrix"], d["dist_coeffs"]

    ok, rvec, tvec = cv2.solvePnP(object_points, img_points, K, dist)
    if not ok:
        print("[エラー] solvePnPが解を見つけられませんでした。点の対応・順序を確認してください。")
        sys.exit(1)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    errors = np.linalg.norm(proj - img_points, axis=1)

    print("\n=== solvePnP 結果(卓球台 -> カメラA) ===")
    print("回転行列 R:\n", R)
    print("並進ベクトル t (mm):\n", t)

    print(f"\n=== 再投影誤差({len(object_points)}点) ===")
    for label, e in zip(labels, errors):
        print(f"  {label}: {e:.2f} px")
    rms = float(np.sqrt(np.mean(errors ** 2)))
    print(f"RMS: {rms:.2f} px")
    if rms > 5.0:
        print("[警告] 再投影誤差が大きめです。クリック位置や実座標にズレがないか確認してください。")

    np.savez(out_path, R=R, t=t)
    print(f"\n保存しました: {out_path}")


if __name__ == "__main__":
    main()