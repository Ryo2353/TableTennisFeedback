"""
Detectball_frame.py が出力した detections.json (カメラA用・カメラB用) を、
offline_sync_check.py と同じ考え方(block_id基準のdelta)でカメラA/B間の
フレームを対応付け、対応が取れたフレームごとに三角測量して卓球台基準の
3D軌跡(X, Y, Z の時系列)を出力する。Phase B-3(バウンドタイミング推定)の
入力データを作るためのスクリプト。

なぜタイムスタンプ(ts)そのものではなくblock_id(bid)のdeltaを使うか:
    カメラAとカメラBのtsは、それぞれのカメラが独立に持つ時計に基づいており、
    ドリフト(実機検証で10秒あたり約78ms)がある。一方、両カメラは同じ
    録画ループから同時に撮影開始しているはずなので、「録画開始から何コマ目か」
    (= bid - そのカメラの最初のbid)は、多少のフレーム落ちがあっても
    両カメラでほぼ一致する。そのためdelta基準でペアリングする。

使い方は2通り:

【新】撮りながら検出(capture_live_detect.py)の結果を使う場合 ← 10/16〜
    python triangulate_trajectory.py <録画フォルダ(rec01など)> <stereo_result.npz> <table_transform.npz> <出力csv>

    録画フォルダの中の camera_a/detections.json と camera_b/detections.json を読む。
    画像は保存していないので、各検出に入っている "delta"(= bid - 録画開始時のbid)を
    そのまま使って対応付ける。

    例:
    python triangulate_trajectory.py \\
        data\\captures\\20261016_190000\\rec01 \\
        results\\20261016\\stereo_result.npz results\\20261016\\table_transform.npz \\
        results\\20261016\\rec01_trajectory.csv

    オプション(新しい使い方のとき):
      --max-reproj 5   再投影誤差がこの画素数より大きい組は捨てる(2台が別の物を見ている組を除く)。
                       0 にすると捨てない。標準は 5。
      --min-z -50      台の座標で Z がこれより低い点は捨てる(台の外に落ちていくボールを除く)。標準は -50mm。
      検出結果に候補が複数("candidates")入っている場合は、2台で一番つじつまの合う(再投影誤差が
      一番小さい)候補の組を自動で選ぶ。

【旧】保存した画像フォルダ + Detectball_frame.py の結果を使う場合
    python triangulate_trajectory.py \
        <カメラA全体フォルダ> <カメラB全体フォルダ> \
        <カメラAのdetections.json> <カメラBのdetections.json> \
        <stereo_result.npz> <table_transform.npz> \
        <出力csv>

例:
    python triangulate_trajectory.py \
        data\\captures\\naka_oku\\camera_a data\\captures\\naka_oku\\camera_b \
        results\\20260918\\ball_detection\\naka_oku_camera_a\\detections.json \
        results\\20260918\\ball_detection\\naka_oku_camera_b\\detections.json \
        results\\20260918\\stereo_result.npz results\\20260918\\table_transform.npz \
        results\\20260918\\ball_detection\\naka_oku_trajectory.csv
"""

import sys
import os
import re
import json
import csv
import numpy as np
import cv2

from triangulate import load_calibration, build_projection_matrices, triangulate, apply_table_transform

BID_TS_PATTERN = re.compile(r"bid(\d+)_ts(\d+)", re.IGNORECASE)


def get_base_bid(folder):
    """そのカメラのセッションフォルダ全体から、最小のbid(=録画開始時点のbid)を得る。"""
    bids = []
    for fname in os.listdir(folder):
        m = BID_TS_PATTERN.search(fname)
        if m:
            bids.append(int(m.group(1)))
    if not bids:
        raise ValueError(f"{folder} にbidを含むファイルが見つかりません。")
    return min(bids)


def load_detections(path, base_bid):
    """旧: 画像フォルダから求めた基準bidで delta を計算する"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for d in data:
        delta = d["bid"] - base_bid
        result[delta] = d
    return result


def load_detections_by_bid(path, base, shift=0):
    """新: カメラの番号(bid)そのもので組にする。
    2台は同期(Bus)で撮っているので、bid が同じコマは同じ瞬間。録画ごとの数え始め(first_bid)には頼らない。
    返す辞書のキーは「bid - base + shift」(base は2台共通の基準。shift はカメラBのずらし用)。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for d in data:
        if d.get("bid") is None:
            continue
        result[d["bid"] - base + shift] = d
    return result


def common_base_bid(info, det_paths):
    """2台共通の基準bid。session_info の first_bid の小さい方(なければ検出の中で一番小さい bid)"""
    if info:
        return min(info["camera_a"]["first_bid"], info["camera_b"]["first_bid"])
    bids = []
    for p in det_paths:
        with open(p, "r", encoding="utf-8") as f:
            bids += [d["bid"] for d in json.load(f) if d.get("bid") is not None]
    return min(bids)


def offset_scan(det_a, det_b, P1, P2, K1, dist1, K2, dist2, R, T_mm, ks=range(-6, 7)):
    """カメラBを k コマずらして組にしたときの、再投影誤差の中央値(各コマの本命の位置で計算)"""
    res = {}
    for k in ks:
        common = sorted(set(det_a) & {d + k for d in det_b})
        if len(common) < 10:
            continue
        p1 = np.array([[det_a[d]["u"], det_a[d]["v"]] for d in common], np.float64)
        p2 = np.array([[det_b[d - k]["u"], det_b[d - k]["v"]] for d in common], np.float64)
        Xc = triangulate(P1, P2, K1, dist1, K2, dist2, p1, p2)
        res[k] = (float(np.median(reproj_errors(Xc, p1, p2, K1, dist1, K2, dist2, R, T_mm))), len(common))
    return res


def pop_opt(argv, flag, default):
    if flag in argv:
        i = argv.index(flag)
        val = float(argv[i + 1])
        del argv[i:i + 2]
        return val
    return default


def candidates_of(d):
    """検出1件から、候補の (u, v) の一覧を返す。candidates が無ければ本命1つだけ。"""
    if d.get("candidates"):
        return [(c[0], c[1]) for c in d["candidates"]]
    return [(d["u"], d["v"])]


def reproj_errors(X_cam_mm, p1, p2, K1, dist1, K2, dist2, R, T_mm):
    """3D点(カメラA基準, mm)を2台の画像に戻したときのずれ(画素)。大きい方を返す。"""
    rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    a, _ = cv2.projectPoints(X_cam_mm, np.zeros(3), np.zeros(3), K1, dist1)
    b, _ = cv2.projectPoints(X_cam_mm, rvec, T_mm, K2, dist2)
    ea = np.linalg.norm(a.reshape(-1, 2) - p1, axis=1)
    eb = np.linalg.norm(b.reshape(-1, 2) - p2, axis=1)
    return np.maximum(ea, eb)


def main():
    argv = sys.argv[:]
    max_reproj = pop_opt(argv, "--max-reproj", 5.0)
    min_z = pop_opt(argv, "--min-z", -50.0)
    b_shift = pop_opt(argv, "--b-shift", 0.0)
    no_auto = "--no-auto-shift" in argv
    if no_auto:
        argv.remove("--no-auto-shift")
    sys.argv = argv
    new_mode = len(sys.argv) == 5
    if not new_mode:
        max_reproj, min_z = 0.0, None   # 旧い使い方では、今まで通り何も捨てない

    if len(sys.argv) == 5:
        # 新: 録画フォルダ(rec01 など)を指定
        rec_dir, stereo_npz_path, table_transform_path, out_csv = sys.argv[1:5]
        det_a_path = os.path.join(rec_dir, "camera_a", "detections.json")
        det_b_path = os.path.join(rec_dir, "camera_b", "detections.json")
        info_path = os.path.join(rec_dir, "session_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            print(f"[情報] カメラA 録画開始時のbid: {info['camera_a']['first_bid']}  "
                  f"コマ数 {info['camera_a']['captured']}  コマ落ち {info['camera_a']['drop_count']}")
            print(f"[情報] カメラB 録画開始時のbid: {info['camera_b']['first_bid']}  "
                  f"コマ数 {info['camera_b']['captured']}  コマ落ち {info['camera_b']['drop_count']}")
        else:
            info = None
        base = common_base_bid(info, [det_a_path, det_b_path])
        det_a = load_detections_by_bid(det_a_path, base)
        det_b = load_detections_by_bid(det_b_path, base, shift=int(b_shift))
        print(f"[情報] 2台の組み方: カメラの番号(bid)が同じコマ同士(基準bid {base}"
              + (f"、カメラBを {int(b_shift):+d} コマずらす" if b_shift else "") + ")")
    elif len(sys.argv) == 8:
        # 旧: 画像フォルダ + detections.json を指定
        folder_a, folder_b, det_a_path, det_b_path, stereo_npz_path, table_transform_path, out_csv = sys.argv[1:8]
        base_a = get_base_bid(folder_a)
        base_b = get_base_bid(folder_b)
        print(f"[情報] カメラA 基準bid: {base_a}")
        print(f"[情報] カメラB 基準bid: {base_b}")
        det_a = load_detections(det_a_path, base_a)
        det_b = load_detections(det_b_path, base_b)
    else:
        print(__doc__)
        sys.exit(1)

    if not det_a or not det_b:
        print("[エラー] どちらかのカメラで検出が1つもありません。")
        sys.exit(1)
    print(f"[情報] カメラA 検出フレーム数: {len(det_a)} (delta範囲: {min(det_a)}〜{max(det_a)})")
    print(f"[情報] カメラB 検出フレーム数: {len(det_b)} (delta範囲: {min(det_b)}〜{max(det_b)})")

    common_deltas = sorted(set(det_a.keys()) & set(det_b.keys()))
    print(f"\n[情報] deltaが一致して対応付けられたフレーム数: {len(common_deltas)}")
    if not common_deltas:
        print("[エラー] 対応が取れたフレームが1つもありません。"
              "bidの飛び方(フレーム落ちのタイミング)がカメラA/Bで大きく違う可能性があります。"
              "±1程度のずれを許容する対応付けを検討してください。")
        sys.exit(1)

    K1, dist1, K2, dist2, R, T = load_calibration(stereo_npz_path)
    P1, P2 = build_projection_matrices(K1, K2, R, T)

    T_arr = np.asarray(T, np.float64).reshape(3)
    T_mm = T_arr * 1000.0 if np.linalg.norm(T_arr) < 10 else T_arr   # stereo_result の T はメートル保存
    print(f"[情報] キャリブレーション: {stereo_npz_path}(2台の距離 {np.linalg.norm(T_mm):.0f}mm)"
          " ← 撮影した日のものか確認してください")

    # 2台のコマの対応がずれていないか、写っているボールの位置から確かめる
    if new_mode:
        scan = offset_scan(det_a, det_b, P1, P2, K1, dist1, K2, dist2, R, T_mm)
        if scan:
            best_k = min(scan, key=lambda k: scan[k][0])
            print("[確認] カメラBをずらしたときの再投影誤差の中央値: "
                  + "  ".join(f"{k:+d}:{scan[k][0]:.1f}" for k in sorted(scan)))
            if best_k != 0 and 0 in scan and scan[best_k][0] < 0.7 * scan[0][0]:
                print(f"[警告] 2台のコマの対応が {best_k:+d} コマずれている可能性があります"
                      f"(中央値 {scan[0][0]:.1f} → {scan[best_k][0]:.1f}画素)。")
                if no_auto:
                    print("       --no-auto-shift が指定されているので、そのまま続けます。")
                else:
                    print(f"       カメラBを {best_k:+d} コマずらして組み直します(やめるには --no-auto-shift)。")
                    det_b = {d + best_k: v for d, v in det_b.items()}
                    common_deltas = sorted(set(det_a.keys()) & set(det_b.keys()))
                    print(f"[情報] 組み直した後のフレーム数: {len(common_deltas)}")
            else:
                print(f"[確認] ずれは見つかりませんでした(一番小さいのは {best_k:+d})。")

    # 各コマで、2台の候補の組み合わせのうち再投影誤差が一番小さいものを選ぶ
    pts1, pts2, errs = [], [], []
    n_multi = 0
    for d in common_deltas:
        ca, cb = candidates_of(det_a[d]), candidates_of(det_b[d])
        if len(ca) > 1 or len(cb) > 1:
            n_multi += 1
        best = None
        for ua in ca:
            for ub in cb:
                q1 = np.array([ua], np.float64)
                q2 = np.array([ub], np.float64)
                Xc = triangulate(P1, P2, K1, dist1, K2, dist2, q1, q2)
                e = float(reproj_errors(Xc, q1, q2, K1, dist1, K2, dist2, R, T_mm)[0])
                if best is None or e < best[2]:
                    best = (ua, ub, e)
        pts1.append(best[0])
        pts2.append(best[1])
        errs.append(best[2])
    pts1 = np.array(pts1, dtype=np.float64)
    pts2 = np.array(pts2, dtype=np.float64)
    errs = np.array(errs)
    if n_multi:
        print(f"[情報] 候補が複数あったコマ: {n_multi}(2台でつじつまの合う組を選びました)")

    estimated_mm = triangulate(P1, P2, K1, dist1, K2, dist2, pts1, pts2)

    td = np.load(table_transform_path)
    estimated_mm = apply_table_transform(estimated_mm, td["R"], td["t"])

    keep = np.ones(len(common_deltas), bool)
    if max_reproj and max_reproj > 0:
        bad = errs > max_reproj
        keep &= ~bad
        print(f"[情報] 再投影誤差 {max_reproj:g}画素より大きい組を除外: {int(bad.sum())} / {len(errs)}"
              f"(2台が別の物を見ていた組)")
    if min_z is not None:
        low = (estimated_mm[:, 2] < min_z) & keep
        keep &= ~low
        print(f"[情報] Z が {min_z:g}mm より低い点を除外: {int(low.sum())}(台の外に落ちていくボールなど)")
    common_deltas = [d for d, k in zip(common_deltas, keep) if k]
    estimated_mm = estimated_mm[keep]
    print(f"[情報] 残った点: {len(common_deltas)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["delta", "ts_a", "ts_b", "x_mm", "y_mm", "z_mm"])
        for i, d in enumerate(common_deltas):
            x, y, z = estimated_mm[i]
            writer.writerow([d, det_a[d]["ts"], det_b[d]["ts"], f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"])

    print(f"\n保存しました: {out_csv}")
    print("\n=== 軌跡(先頭10点) ===")
    for i, d in enumerate(common_deltas[:10]):
        x, y, z = estimated_mm[i]
        print(f"  delta={d:5d}  X={x:8.1f}  Y={y:8.1f}  Z={z:8.1f}")


if __name__ == "__main__":
    main()