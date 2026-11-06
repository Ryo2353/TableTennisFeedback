# -*- coding: utf-8 -*-
r"""
撮影の後の処理を1回の操作でまとめる。録画フォルダを渡すと、次を順に動かす。

    track_balls.py(追跡)→ landings_from_tracks.py(着地点)→ project_heatmap.py(ヒートマップ)

1セットに録画が複数あるときは、録画フォルダを並べて渡す。着地点は1つにまとめ、ヒートマップは1枚になる。
ここでは計算をせず、今のスクリプトを順に呼ぶだけ(結果は1つずつ手で動かしたときと同じ)。

使い方(プロジェクトのいちばん上のフォルダで):
    python src/run_set.py <録画フォルダ> [<録画フォルダ> ...] --calib <キャリブレーションのフォルダ> --projector <projector_transform.npz>

例(撮影の現場で、台に映す):
    python src/run_set.py data/captures/20261031/20261031_120000/rec03 --calib results/20261031 --projector results/20261031/projector_transform.npz

例(家で、画像だけ作る):
    python src/run_set.py data/captures/20261031/20261031_120000/rec03 --calib results/20261031 --no-show

指定するもの:
    --calib      stereo_result.npz と table_transform.npz が入ったフォルダ(必ず指定する)
                 設置のたびに結果が変わるので、古い結果を間違って使わないよう毎回指定する。
    --projector  projector_transform.npz。省略すると投影せず、台の座標のままの画像だけを保存する。
    --no-show    画面に映さない(画像の保存だけ)。
    --out        結果を入れるフォルダ。省略すると <--calib のフォルダ>/set_日付_時刻 を毎回新しく作るので、
                 前の結果を上書きしない。
    --table      台の名前(src/table_config.py)。省略すると今の設定。
    --points, --outline, --bandwidth   project_heatmap.py にそのまま渡す。

出力(--out のフォルダの中):
    <録画名>_tracks.csv / .png    追跡の結果と確認用の図
    <録画名>_tracks_landings.csv  その録画の着地点
    set_landings.csv              セット全体の着地点(1列目 rec に録画名が入る)
    heatmap.png                   プロジェクターに映す画像(--projector を指定したときだけ)
    heatmap_table.png             台の座標のままの画像

途中で失敗した段階があれば、そこで止める。録画を1つ抜かしたままのヒートマップを映さないため。
"""

import argparse
import csv
import os
import subprocess
import sys
import time

from table_config import get_table

SRC = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.join(SRC, "triangulation", "track_balls.py")
LANDINGS = os.path.join(SRC, "triangulation", "landings_from_tracks.py")
HEATMAP = os.path.join(SRC, "projector", "project_heatmap.py")


def fail(msg):
    print(f"[エラー] {msg}")
    sys.exit(1)


def run_step(name, cmd):
    """スクリプトを1つ動かす。失敗したらそこで止める"""
    print(f"\n===== {name} =====")
    if subprocess.run(cmd).returncode != 0:
        fail(f"「{name}」で止まりました。上のメッセージを確認してください。")


def rec_names(rec_dirs):
    """録画ごとの名前。同じ名前(別の日の rec02 など)があれば、1つ上のフォルダ名を前に付ける"""
    tails = [os.path.basename(os.path.normpath(r)) for r in rec_dirs]
    names = []
    for r, t in zip(rec_dirs, tails):
        if tails.count(t) > 1:
            parent = os.path.basename(os.path.dirname(os.path.abspath(r)))
            t = f"{parent}_{t}"
        names.append(t)
    if len(set(names)) != len(names):
        fail("同じ録画フォルダが2回指定されています。")
    return names


def check_calib(calib, projector, table_name):
    """台の名前とキャリブレーションのファイルを確かめる(capture_live_detect.py も撮影の前に使う)"""
    get_table(table_name)
    stereo = os.path.join(calib, "stereo_result.npz")
    table = os.path.join(calib, "table_transform.npz")
    for p in (stereo, table):
        if not os.path.isfile(p):
            fail(f"キャリブレーションのファイルがありません: {p}")
    if projector and not os.path.isfile(projector):
        fail(f"プロジェクターの位置合わせのファイルがありません: {projector}")
    return stereo, table


def check_inputs(a):
    """動かし始める前に、台の名前とファイルがそろっているか全部確かめる"""
    stereo, table = check_calib(a.calib, a.projector, a.table)
    for r in a.rec_dirs:
        for cam in ("camera_a", "camera_b"):
            p = os.path.join(r, cam, "detections.json")
            if not os.path.isfile(p):
                fail(f"検出結果がありません: {p}")
    return stereo, table


def merge_landings(per_rec, out_csv):
    """録画ごとの着地点を1つにまとめる。録画ごとの個数を返す"""
    counts = []
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rec", "track", "t_s", "x_mm", "y_mm", "z_min_mm"])
        for name, path in per_rec:
            with open(path, encoding="utf-8", newline="") as g:
                rows = list(csv.DictReader(g))
            for r in rows:
                w.writerow([name, r["track"], r["t_s"], r["x_mm"], r["y_mm"], r["z_min_mm"]])
            counts.append((name, len(rows)))
    return counts


def main():
    ap = argparse.ArgumentParser(description="録画フォルダから、追跡 → 着地点 → ヒートマップまでを続けて動かす")
    ap.add_argument("rec_dirs", nargs="+", help="録画フォルダ(camera_a, camera_b の入ったもの)。複数指定できる")
    ap.add_argument("--calib", required=True, help="stereo_result.npz と table_transform.npz が入ったフォルダ")
    ap.add_argument("--projector", default=None, help="projector_transform.npz。省略すると投影しない")
    ap.add_argument("--no-show", action="store_true", help="画面に映さない(画像の保存だけ)")
    ap.add_argument("--out", default=None, help="結果を入れるフォルダ")
    ap.add_argument("--table", default=None, help="台の名前(src/table_config.py)。省略すると今の設定")
    ap.add_argument("--points", action="store_true", help="着地点そのものも重ねて映す")
    ap.add_argument("--outline", action="store_true", help="計測範囲の枠を映す")
    ap.add_argument("--bandwidth", type=float, default=60.0, help="1点をぼかす広さ(mm)")
    a = ap.parse_args()

    t_start = time.time()
    stereo, table = check_inputs(a)
    names = rec_names(a.rec_dirs)
    out_dir = a.out or os.path.join(a.calib, time.strftime("set_%Y%m%d_%H%M%S"))
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        fail(f"結果のフォルダがすでに使われています: {out_dir}(上書きしないため止めました)")
    os.makedirs(out_dir, exist_ok=True)
    print(f"録画 {len(a.rec_dirs)} 個 / キャリブレーション: {a.calib} / 結果: {out_dir}")

    table_opt = ["--table", a.table] if a.table else []
    per_rec = []
    for name, rec in zip(names, a.rec_dirs):
        tracks = os.path.join(out_dir, f"{name}_tracks.csv")
        landings = os.path.join(out_dir, f"{name}_tracks_landings.csv")
        run_step(f"{name}: 追跡", [sys.executable, TRACK, rec, stereo, table, tracks] + table_opt)
        run_step(f"{name}: 着地点", [sys.executable, LANDINGS, tracks, landings] + table_opt)
        per_rec.append((name, landings))

    set_csv = os.path.join(out_dir, "set_landings.csv")
    counts = merge_landings(per_rec, set_csv)
    total = sum(n for _, n in counts)
    print("\n===== セット全体の着地点 =====")
    for name, n in counts:
        print(f"  {name}: {n} 個")
    print(f"  合計: {total} 個 → {set_csv}")
    if total == 0:
        print("[警告] 着地点が1つもありません。ヒートマップは真っ黒になります。")

    cmd = [sys.executable, HEATMAP, set_csv, "--save", os.path.join(out_dir, "heatmap.png"),
           "--bandwidth", str(a.bandwidth)] + table_opt
    if a.points:
        cmd.append("--points")
    if a.outline:
        cmd.append("--outline")
    if a.projector:
        cmd += ["--transform", a.projector]
    else:
        print("\n[情報] --projector がないので投影せず、台の座標のままの画像だけを保存します。")
    if a.no_show or not a.projector:
        cmd.append("--no-show")
    print(f"\n[時間] 追跡と着地点の処理: {time.time() - t_start:.1f} 秒")
    run_step("ヒートマップ", cmd)
    print(f"\n完了: {out_dir}")


if __name__ == "__main__":
    main()
