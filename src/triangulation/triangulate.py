"""
三角測量コアモジュール

stereo_calibrate_charuco.py が保存した stereo_result.npz(カメラ1・2の内部
パラメータ、外部パラメータR・T)を読み込み、両カメラで検出したボールの
画像上の座標(u,v)ペアから、実世界の3次元座標(X,Y,Z、単位mm)を計算する。

単体では実行しない(python3 triangulate.py で自己テストのみ動く)。
実際のデータに使うときは triangulate_points.py から呼び出す。
"""

import numpy as np
import cv2


def load_calibration(stereo_npz_path):
    """stereo_result.npz を読み込み、必要な行列一式を返す。

    stereo_calibrate_charuco.py は CALIB_FIX_INTRINSIC で計算しているため、
    stereo_result.npz 単体に両カメラの内部パラメータも含まれている
    (camera_a/camera_b それぞれの calibration_result.npz を別途読む必要はない)。

    注意: T はChArUcoボードをメートル単位(square_mm/1000)で定義して求めた
    ものなので、メートル単位で保存されている。
    """
    d = np.load(stereo_npz_path)
    K1, dist1 = d["camera_matrix1"], d["dist1"]
    K2, dist2 = d["camera_matrix2"], d["dist2"]
    R, T = d["R"], d["T"]
    return K1, dist1, K2, dist2, R, T


def build_projection_matrices(K1, K2, R, T):
    """カメラ1を基準(原点・無回転)とした射影行列 P1・P2 を組み立てる。

    P1・P2はキャリブレーション結果が変わらない限り不変なので、
    フレームごとに計算し直す必要はなく、最初に1回だけ呼べばよい。
    """
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T.reshape(3, 1)])
    return P1, P2


def triangulate(P1, P2, K1, dist1, K2, dist2, pts1, pts2):
    """両カメラの画像上の点(pts1, pts2)から3次元座標を計算する。

    引数:
        pts1, pts2: (N, 2) の配列。各カメラで検出したボールの(u, v)ピクセル座標。
                    同じ行同士が「同じ瞬間」に対応している必要がある。
    戻り値:
        (N, 3) の配列。実世界座標(X, Y, Z)、単位はmm。
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 1, 2)

    # レンズ歪みを補正する(P=K を渡すことで、補正後も元のピクセル座標系に戻す)
    und1 = cv2.undistortPoints(pts1, K1, dist1, P=K1).reshape(-1, 2)
    und2 = cv2.undistortPoints(pts2, K2, dist2, P=K2).reshape(-1, 2)

    # triangulatePoints は (2, N) 形状・同次座標(4, N)を要求する
    points_4d = cv2.triangulatePoints(P1, P2, und1.T, und2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T  # (N, 3)、単位はメートル

    return points_3d * 1000.0  # mmに変換して返す


def apply_table_transform(points_cam_mm, R, t):
    """カメラ基準の3D座標を、卓球台基準の3D座標に変換する。

    R, t は compute_table_pose.py が cv2.solvePnP で求めたもので、
    「卓球台座標 -> カメラ座標」の変換(X_cam = R @ X_table + t)。
    ここではその逆変換(X_table = R^T @ (X_cam - t))を適用する。

    引数:
        points_cam_mm: (N, 3) カメラ基準の3D座標(mm)。triangulate()の戻り値。
        R: (3, 3) 回転行列(卓球台->カメラ)
        t: (3,) または (3,1) 並進ベクトル(mm、卓球台->カメラ)
    戻り値:
        (N, 3) 卓球台基準の3D座標(mm)
    """
    points_cam_mm = np.asarray(points_cam_mm, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return (points_cam_mm - t) @ R  # R^T @ (X - t) を行ごとに適用(R @ x の転置形)


def _self_test():
    """合成データで、既知の3次元点が正しく復元できるか確認する自己テスト。"""
    rng = np.random.default_rng(0)

    K1 = np.array([[1200, 0, 640], [0, 1200, 480], [0, 0, 1]], dtype=np.float64)
    K2 = np.array([[1300, 0, 660], [0, 1300, 500], [0, 0, 1]], dtype=np.float64)
    dist1 = np.zeros(5)
    dist2 = np.zeros(5)

    # カメラ2は、カメラ1に対して0.75m離れ、少し回転しているとする
    angle = np.deg2rad(35)
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    T = np.array([[0.75], [0.02], [0.10]])  # メートル単位(npzの実際の保存形式に合わせる)

    P1, P2 = build_projection_matrices(K1, K2, R, T)

    # ランダムな3次元点(カメラの前方、単位メートル)を用意し、両カメラに投影する
    true_points_m = rng.uniform([-0.3, -0.3, 1.0], [0.3, 0.3, 2.0], size=(10, 3))

    def project(P, K, dist, pts_m):
        # P = K [R|t] なので、Rとtを取り出してprojectPointsで投影する
        Rmat = np.linalg.inv(K) @ P[:, :3]
        tvec = (np.linalg.inv(K) @ P[:, 3]).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(Rmat)
        pix, _ = cv2.projectPoints(pts_m, rvec, tvec, K, dist)
        return pix.reshape(-1, 2)

    pts1 = project(P1, K1, dist1, true_points_m)
    pts2 = project(P2, K2, dist2, true_points_m)

    estimated_mm = triangulate(P1, P2, K1, dist1, K2, dist2, pts1, pts2)
    true_mm = true_points_m * 1000.0

    error = np.linalg.norm(estimated_mm - true_mm, axis=1)
    print("自己テスト: 合成データでの復元誤差(mm)")
    print(error)
    assert np.allclose(estimated_mm, true_mm, atol=1e-3), "三角測量の自己テストに失敗しました"
    print("自己テスト成功: 誤差はすべて無視できるレベル(浮動小数点誤差のみ)です")


def _self_test_table_transform():
    """apply_table_transform()の自己テスト。

    「卓球台上の既知点」を、適当なカメラ姿勢で「カメラ基準の座標」に変換し、
    その変換のR,tを使ってapply_table_transform()で卓球台基準に戻せるか確認する。
    """
    rng = np.random.default_rng(1)

    # 卓球台->カメラの姿勢を適当に決める(実際はsolvePnPで求める値に相当)
    angle = np.deg2rad(20)
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    t = np.array([300.0, -50.0, 1500.0])  # mm

    true_table_mm = rng.uniform([0, 0, 0], [1165, 840, 300], size=(10, 3))
    cam_mm = true_table_mm @ R.T + t  # X_cam = R @ X_table + t (行ベクトル版)

    recovered_table_mm = apply_table_transform(cam_mm, R, t)

    error = np.linalg.norm(recovered_table_mm - true_table_mm, axis=1)
    print("\n自己テスト: 卓球台座標変換の復元誤差(mm)")
    print(error)
    assert np.allclose(recovered_table_mm, true_table_mm, atol=1e-6), "卓球台座標変換の自己テストに失敗しました"
    print("自己テスト成功: apply_table_transform()は正しく逆変換できています")


if __name__ == "__main__":
    _self_test()
    _self_test_table_transform()