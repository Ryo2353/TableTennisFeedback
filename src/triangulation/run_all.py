"""
data/captures/hitball1 〜 hitball10 のそれぞれに対して、
  1. Detectball_frame.py (camera_a, camera_b)
  2. triangulate_trajectory.py
  3. fit_bounce.py
を順番に自動実行し、有効と判定された着地点をすべて1つのCSVに集約する
(Phase C-1: 複数打球の着地点を集める)。

前提:
    - data/captures/hitball1 〜 hitball10 に camera_a/, camera_b/ フォルダがあり、
      capture_dual.py で撮影した*.bmp画像が入っていること
    - results/20260918/stereo_result.npz, results/20260918/table_transform.npz
      が作成済みであること

使い方:
    python run_all.py

出力:
    results/20260918/ball_detection/<hitball名>_camera_a,b/detections.json
    results/20260918/ball_detection/<hitball名>_trajectory.csv
    results/20260918/ball_detection/<hitball名>_trajectory_landings.csv
    results/20260918/ball_detection/all_landings.csv  (全打球の着地点をまとめたもの)
"""

import os
import csv
import subprocess

HITBALLS = [f"hitball{i}" for i in range(1, 11)]

BASE_DATA = r"data\captures"
BASE_RESULTS = r"results\20260918\ball_detection"
STEREO_NPZ = r"results\20260918\stereo_result.npz"
TABLE_TRANSFORM = r"results\20260918\table_transform.npz"

DETECT_SCRIPT = r"src\ball_detection\Detectball_frame.py"
TRAJ_SCRIPT = r"src\triangulation\triangulate_trajectory.py"
BOUNCE_SCRIPT = r"src\triangulation\fit_bounce.py"


def run(cmd):
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(BASE_RESULTS, exist_ok=True)
    all_landings = []
    failed = []

    for hb in HITBALLS:
        print(f"\n========== {hb} ==========", flush=True)
        folder_a = os.path.join(BASE_DATA, hb, "camera_a")
        folder_b = os.path.join(BASE_DATA, hb, "camera_b")

        if not os.path.isdir(folder_a) or not os.path.isdir(folder_b):
            print(f"  [スキップ] データフォルダが見つかりません: {folder_a} / {folder_b}")
            failed.append((hb, "フォルダなし"))
            continue

        out_a = os.path.join(BASE_RESULTS, f"{hb}_camera_a")
        out_b = os.path.join(BASE_RESULTS, f"{hb}_camera_b")
        traj_csv = os.path.join(BASE_RESULTS, f"{hb}_trajectory.csv")

        try:
            run(["python", DETECT_SCRIPT, folder_a, out_a])
            run(["python", DETECT_SCRIPT, folder_b, out_b])

            det_a = os.path.join(out_a, "detections.json")
            det_b = os.path.join(out_b, "detections.json")

            run(["python", TRAJ_SCRIPT, folder_a, folder_b, det_a, det_b,
                 STEREO_NPZ, TABLE_TRANSFORM, traj_csv])

            run(["python", BOUNCE_SCRIPT, traj_csv])
        except subprocess.CalledProcessError as e:
            print(f"  [失敗] {hb} の処理中にエラーが発生しました: {e}")
            failed.append((hb, str(e)))
            continue

        landings_csv = traj_csv.rsplit(".csv", 1)[0] + "_landings.csv"
        if os.path.exists(landings_csv):
            with open(landings_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_landings.append({
                        "hitball": hb,
                        "x_mm": row["x_mm"],
                        "y_mm": row["y_mm"],
                        "z_min_mm": row["z_min_mm"],
                    })

    out_path = os.path.join(BASE_RESULTS, "all_landings.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hitball", "x_mm", "y_mm", "z_min_mm"])
        writer.writeheader()
        writer.writerows(all_landings)

    print(f"\n\n=== 全体まとめ ===")
    print(f"有効な着地点: {len(all_landings)}件")
    for row in all_landings:
        print(f"  {row['hitball']}: X={row['x_mm']}mm, Y={row['y_mm']}mm")
    if failed:
        print(f"\n処理できなかったフォルダ: {len(failed)}件")
        for hb, reason in failed:
            print(f"  {hb}: {reason}")
    print(f"\n保存しました: {out_path}")


if __name__ == "__main__":
    main()