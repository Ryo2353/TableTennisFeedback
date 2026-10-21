import numpy as np
np.set_printoptions(precision=4, suppress=True)

single = np.load(r"data\captures\calibrate\camera_a\calibration_result.npz")
stereo = np.load(r"results\20260918\stereo_result.npz")

print("stereo_result.npz のキー:", list(stereo.keys()))
print("\n--- 単眼(calibration_result.npz) ---")
print("K:\n", single["camera_matrix"])
print("dist:", single["dist_coeffs"].ravel())

print("\n--- ステレオ(stereo_result.npz)のカメラA ---")
for key in stereo.keys():
    if key.upper() in ("K1", "CAMERA_MATRIX1", "M1", "DIST1", "D1", "DIST_COEFFS1"):
        print(f"{key}:\n", stereo[key])