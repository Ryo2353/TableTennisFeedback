import numpy as np
np.set_printoptions(precision=3, suppress=True)

td = np.load(r"results\20260918\table_transform.npz")   # 精度検証で使ったもの
st = np.load(r"results\20260918\stereo_result.npz")                       # 精度検証で使ったもの

R, t = td["R"], td["t"]
C = -R.T @ t                          # カメラAの位置(卓球台座標, mm)
view = R.T @ np.array([0, 0, 1.0])    # カメラAの視線の向き(卓球台座標)

bias = np.array([5.47, -12.21, -16.40])   # 精度検証の平均のずれ
cos = abs(bias @ view) / np.linalg.norm(bias)

print("カメラAの位置 (mm):", C)
print("カメラAの視線の向き:", view)
print("ずれと視線の向きの角度 (度):", round(float(np.degrees(np.arccos(cos))), 1))
print("カメラ間の距離 |T| (m):", round(float(np.linalg.norm(st["T"])), 4))