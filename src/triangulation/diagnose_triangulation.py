# -*- coding: utf-8 -*-
"""
3D位置がおかしいときの原因を切り分けるための診断スクリプト。

triangulate_trajectory.py と同じ方法で、録画フォルダ(rec01 など)の検出結果を
2台で組にして3D位置を計算し、さらに次の2つを調べる。

  1. 再投影誤差(画素)
     計算した3D位置を、もう一度カメラA・Bの画像に戻して、検出した位置とどれだけ
     ずれるかを見る。
       - 小さい(数画素以内) → 2台が同じボールを、同じ瞬間に見ている。
                               3D位置(カメラA基準)は正しい。おかしいのは「台の位置(C)」の方。
       - 大きい(数十画素以上) → 2台の組み合わせ(同じ瞬間・同じ物か)か、
                               2台の位置関係(B)がおかしい。

  2. 台の座標に直したときの Z(高さ)
     ボールは台の下に行けないので、Z が大きくマイナスの点が多ければ、
     台の位置(C)か、座標の決め方(4隅のクリック順など)がおかしい。

  ※ 再投影誤差は、2台の組み合わせの間違いのうち「エピポーラ線から外れる方向」のずれしか
     拾えない。誤差が小さくても、組み合わせが絶対に正しいとまでは言えない。

使い方(triangulate_trajectory.py と同じフォルダで):
    python diagnose_triangulation.py <録画フォルダ> <stereo_result.npz> <table_transform.npz>

例:
    python src\\triangulation\\diagnose_triangulation.py data\\captures\\20261016_190114\\rec02 results\\20261016\\stereo_result.npz results\\20261016\\table_transform.npz
"""

import json
import os
import sys

import cv2
import numpy as np

from triangulate import load_calibration, build_projection_matrices, triangulate, apply_table_transform


def load_det(path):
    with open(path, encoding="utf-8") as f:
        return {d["delta"]: d for d in json.load(f) if d.get("delta") is not None}


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    rec_dir, stereo_path, table_path = sys.argv[1:4]

    det_a = load_det(os.path.join(rec_dir, "camera_a", "detections.json"))
    det_b = load_det(os.path.join(rec_dir, "camera_b", "detections.json"))
    common = sorted(set(det_a) & set(det_b))
    print(f"組になったコマ: {len(common)}")
    if not common:
        sys.exit(1)

    K1, dist1, K2, dist2, R, T = load_calibration(stereo_path)
    P1, P2 = build_projection_matrices(K1, K2, R, T)
    pts1 = np.array([[det_a[d]["u"], det_a[d]["v"]] for d in common], np.float64)
    pts2 = np.array([[det_b[d]["u"], det_b[d]["v"]] for d in common], np.float64)
    X_cam = triangulate(P1, P2, K1, dist1, K2, dist2, pts1, pts2)   # mm, カメラA基準

    # ---- 1. 再投影誤差 ----
    T = np.asarray(T, np.float64).reshape(3)
    T_mm = T * 1000.0 if np.linalg.norm(T) < 10 else T   # stereo_result の T はメートル保存
    rvec_b, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    proj_a, _ = cv2.projectPoints(X_cam, np.zeros(3), np.zeros(3), K1, dist1)
    proj_b, _ = cv2.projectPoints(X_cam, rvec_b, T_mm, K2, dist2)
    err_a = np.linalg.norm(proj_a.reshape(-1, 2) - pts1, axis=1)
    err_b = np.linalg.norm(proj_b.reshape(-1, 2) - pts2, axis=1)
    err = np.maximum(err_a, err_b)

    print("\n=== 1. 再投影誤差(画素) ===")
    print(f"  中央値 {np.median(err):.1f}   90%点 {np.percentile(err, 90):.1f}   最大 {err.max():.1f}")
    for th in (3, 10, 30):
        print(f"  {th}画素以下のコマ: {int((err <= th).sum())} / {len(err)}")
    print(f"  カメラAからの距離(mm): 中央値 {np.median(np.linalg.norm(X_cam, axis=1)):.0f}"
          f"  (参考: 台の原点までの距離は table_transform の t の長さ)")

    # ---- 2. 台の座標での Z ----
    td = np.load(table_path)
    X_tab = apply_table_transform(X_cam, td["R"], td["t"])
    print(f"  台の原点までの距離(mm): {np.linalg.norm(td['t']):.0f}")
    good = err <= 10
    print("\n=== 2. 台の座標での位置(再投影誤差10画素以下のコマだけ) ===")
    if good.sum() < 3:
        print("  使えるコマが少なすぎます。1の結果から、2台の組み合わせかステレオの問題を疑います。")
    else:
        Xg = X_tab[good]
        print(f"  X: {Xg[:, 0].min():.0f} 〜 {Xg[:, 0].max():.0f} mm")
        print(f"  Y: {Xg[:, 1].min():.0f} 〜 {Xg[:, 1].max():.0f} mm")
        print(f"  Z: {Xg[:, 2].min():.0f} 〜 {Xg[:, 2].max():.0f} mm   (正しければ、だいたい 0〜数百mm)")
        below = int((Xg[:, 2] < -50).sum())
        print(f"  台より50mm以上下(Z < -50mm)の点: {below} / {len(Xg)}"
              "  (ボールは台の下に行けないので、多ければ台の位置(C)か座標の決め方がおかしい)")
        # 台の上のどのあたりで Z がおかしいか(X方向に3つに分けて、Zの最小値を見る)
        edges = np.linspace(Xg[:, 0].min(), Xg[:, 0].max(), 4)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (Xg[:, 0] >= lo) & (Xg[:, 0] <= hi)
            if m.any():
                print(f"    X {lo:6.0f}〜{hi:6.0f}mm: Zの最小 {Xg[m, 2].min():7.0f}mm ({int(m.sum())}点)")

    # ---- 3. 組み合わせのずらし方を変えて、再投影誤差がどう変わるか ----
    # カメラBの delta を k だけずらして組にし直す。2台の数え始めが k コマずれていれば、
    # その k のときに再投影誤差が一番小さくなる。
    print("\n=== 3. カメラBを k コマずらして組にしたときの再投影誤差(中央値) ===")
    rvec_b2 = rvec_b
    best = None
    for k in range(-6, 7):
        pairs = [d for d in sorted(det_a) if (d + k) in det_b]
        if len(pairs) < 10:
            continue
        q1 = np.array([[det_a[d]["u"], det_a[d]["v"]] for d in pairs], np.float64)
        q2 = np.array([[det_b[d + k]["u"], det_b[d + k]["v"]] for d in pairs], np.float64)
        Xk = triangulate(P1, P2, K1, dist1, K2, dist2, q1, q2)
        pa_k, _ = cv2.projectPoints(Xk, np.zeros(3), np.zeros(3), K1, dist1)
        pb_k, _ = cv2.projectPoints(Xk, rvec_b2, T_mm, K2, dist2)
        e_k = np.maximum(np.linalg.norm(pa_k.reshape(-1, 2) - q1, axis=1),
                         np.linalg.norm(pb_k.reshape(-1, 2) - q2, axis=1))
        med = float(np.median(e_k))
        n_good = int((e_k <= 5).sum())
        mark = ""
        if best is None or med < best[1]:
            best = (k, med)
        print(f"  k={k:+d}: 組 {len(pairs):4d}  中央値 {med:7.1f}画素  5画素以下 {n_good:4d}")
    if best is not None:
        print(f"  → 一番小さいのは k={best[0]:+d}(中央値 {best[1]:.1f}画素)")
        if best[0] != 0 and best[1] < 5:
            print(f"     2台の数え始めが {best[0]:+d} コマずれている可能性が高い。")
        elif best[1] >= 10:
            print("     どのずらし方でも誤差が大きい → 2台が別の物を見ているか、ステレオ(B)がおかしい。")

    print("\n=== 判定の目安 ===")
    if np.median(err) <= 3:
        print("  再投影誤差は小さい → 2台の組み合わせ・ステレオは正しい。台の位置(C)を疑う。")
    elif np.median(err) <= 10:
        print("  再投影誤差はやや大きい → 同期のずれ(1コマ未満)や検出位置のぶれの可能性。")
    else:
        print("  再投影誤差が大きい → 2台が別の物を見ている/別の瞬間を組にしている/ステレオがおかしい。")

    # コマごとの詳細をCSVに出す(家で見返す用)
    out = os.path.join(rec_dir, "diagnose_points.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("delta,u_a,v_a,u_b,v_b,err_a_px,err_b_px,x_mm,y_mm,z_mm\n")
        for i, d in enumerate(common):
            f.write(f"{d},{pts1[i,0]:.1f},{pts1[i,1]:.1f},{pts2[i,0]:.1f},{pts2[i,1]:.1f},"
                    f"{err_a[i]:.1f},{err_b[i]:.1f},{X_tab[i,0]:.1f},{X_tab[i,1]:.1f},{X_tab[i,2]:.1f}\n")
    print(f"\nコマごとの詳細: {out}")


if __name__ == "__main__":
    main()