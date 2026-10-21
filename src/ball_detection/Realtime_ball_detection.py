"""
横カメラ(または真上カメラ)からリアルタイムで映像を取得し、
既存の検出ロジック(3フレーム差分 + Contourベースのボール判定)を
そのまま適用して動作確認するためのスクリプト。

元になったコード(連番画像を処理するもの)との違いは、
「① 入力元がファイルではなくpytelicam経由のライブ映像」
「② 取得(capture)にかかった時間と検出(processing)にかかった時間を
   分けて計測し、一定間隔で表示する」の2点のみ。
検出ロジック自体(3フレーム差分・二値化・輪郭検出・フィルタ条件)は
元のコードと同じものを使っている。

前提:
    pip install pytelicam opencv-python numpy

注意:
    露光時間・ゲイン・ROIなどのカメラパラメータ設定、
    Mono8(白黒)出力が本当に "pytelicam.OutputImageType.Mono8" という
    名前で提供されているかは、お使いのpytelicamのバージョンで
    差異がある可能性があります。
    実行前に一度、以下のコマンドで実際に使えるメンバー名を
    確認しておくことを推奨します:

        python -c "import pytelicam; print([x for x in dir(pytelicam.OutputImageType)])"
        python -c "import pytelicam; cs=pytelicam.get_camera_system(); d=cs.create_device_object(); d.open(); print(dir(d))"

    もし名前が違っていた場合は、このファイルの CAM_* 定数や
    get_ndarray() の引数を実際の名前に置き換えてください。
"""

import collections
import time

import cv2
import numpy as np
import pytelicam

# ============================================================
# 設定(元のコードから引き継いだ検出パラメータ)
# ============================================================
MIN_AREA = 100
MAX_AREA = 4000
MIN_CIRCULARITY = 0.3
MIN_BRIGHTNESS = 150

# 何フレームごとに速度サマリーを表示するか
REPORT_INTERVAL = 100

# カメラの露光時間・ゲイン設定(フル解像度1440x1080時の仕様値)
EXPOSURE_TIME_US = 1000.0  # 露光時間(マイクロ秒)
GAIN_DB = 20.0              # ゲイン(dB)

# 目視確認用のプレビューウィンドウを表示するか
# True にすると、検出処理に画面表示の時間が追加でかかるため、fpsはさらに落ちます。
# あくまで「ちゃんと検出できているか」を目で見て確認するためのモードです。
SHOW_PREVIEW = True
PREVIEW_WINDOW_NAME = "ball detection preview (qで終了)"

# 出力フォーマット。お使いのpytelicamで Mono8 が使えない場合は
# Bgr24 に変えて、後段で cv2.cvtColor(..., COLOR_BGR2GRAY) してください。
OUTPUT_IMAGE_TYPE_NAME = "Bgr24"


def open_camera():
    """pytelicamでカメラを開き、ストリームを開始して device を返す"""
    cam_system = pytelicam.get_camera_system()
    cam_num = cam_system.get_num_of_cameras()
    if cam_num == 0:
        raise RuntimeError("カメラが見つかりませんでした(接続・ドライバを確認してください)")
    print(f"接続カメラ台数: {cam_num}")

    cam_device = cam_system.create_device_object()
    cam_device.open()

    # 露光時間・ゲインを仕様値に設定(GenAPI経由)
    # ノード名は AcquisitionControl > ExposureTime, AnalogControl > Gain (どちらもFloat型)
    try:
        cam_device.genapi.set_float_value("ExposureTime", EXPOSURE_TIME_US)
        cam_device.genapi.set_float_value("Gain", GAIN_DB)
        print(f"露光時間: {EXPOSURE_TIME_US}us, ゲイン: {GAIN_DB}dB に設定しました")
    except Exception as e:
        print(f"警告: 露光時間・ゲインの設定に失敗しました({e})。デフォルト設定のまま続行します。")

    cam_device.cam_stream.open()
    cam_device.cam_stream.start()
    return cam_system, cam_device


def close_camera(cam_system, cam_device):
    cam_device.cam_stream.stop()
    cam_device.cam_stream.close()
    cam_device.close()
    cam_system.terminate()


def grab_gray_frame(cam_device, output_type):
    """1フレーム取得し、グレースケールのnumpy配列として返す"""
    image_data = cam_device.cam_stream.get_next_image()
    try:
        np_arr = image_data.get_ndarray(output_type)
    finally:
        image_data.release()

    if np_arr.ndim == 3:
        # Bgr24等、カラーで来た場合はグレースケールに変換
        np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGR2GRAY)
    return np_arr


def detect_ball_candidates(prev_frame, frame, next_frame):
    """元のコードと同じ、3フレーム差分 + Contourベースの判定"""
    diff1 = cv2.absdiff(prev_frame, frame)
    diff2 = cv2.absdiff(frame, next_frame)
    diff = cv2.bitwise_and(diff1, diff2)

    _, binary = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        roi = frame[max(0, int(cy) - 5):min(frame.shape[0], int(cy) + 5),
                    max(0, int(cx) - 5):min(frame.shape[1], int(cx) + 5)]
        brightness = np.mean(roi)
        if brightness < MIN_BRIGHTNESS:
            continue

        candidates.append((cx, cy, area, circularity, brightness))

    return candidates


def main():
    output_type = getattr(pytelicam.OutputImageType, OUTPUT_IMAGE_TYPE_NAME)
    cam_system, cam_device = open_camera()

    # 直近3フレームを保持するスライディングウィンドウ
    frame_buffer = collections.deque(maxlen=3)

    frame_count = 0
    detected_count = 0

    # 速度計測用(直近REPORT_INTERVAL件分をためて平均を出す)
    capture_times = collections.deque(maxlen=REPORT_INTERVAL)
    process_times = collections.deque(maxlen=REPORT_INTERVAL)

    print("リアルタイム検出を開始します。Ctrl+Cで終了します。")
    session_start = time.time()

    try:
        while True:
            # ---- ① カメラからの取得時間を計測 ----
            t0 = time.time()
            frame = grab_gray_frame(cam_device, output_type)
            t1 = time.time()
            capture_times.append(t1 - t0)

            frame_buffer.append(frame)
            frame_count += 1

            if len(frame_buffer) < 3:
                continue  # 3枚たまるまでは検出できない

            prev_frame, mid_frame, next_frame = frame_buffer

            # ---- ② 検出処理の時間を計測 ----
            t2 = time.time()
            candidates = detect_ball_candidates(prev_frame, mid_frame, next_frame)
            t3 = time.time()
            process_times.append(t3 - t2)

            if candidates:
                detected_count += 1
                # 必要な場合だけ有効化(頻度が高いと表示自体が重くなるので注意)
                # print(", ".join([f"({int(c[0])},{int(c[1])})" for c in candidates]))

            if SHOW_PREVIEW:
                preview = cv2.cvtColor(mid_frame, cv2.COLOR_GRAY2BGR)
                for cx, cy, area, circularity, brightness in candidates:
                    bx, by = int(cx), int(cy)
                    cv2.circle(preview, (bx, by), 15, (0, 0, 255), 2)
                    cv2.putText(preview, f"({bx},{by})", (bx + 10, by - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(preview, f"frame {frame_count} / detected {detected_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(PREVIEW_WINDOW_NAME, preview)
                # 'q' キーで終了できるようにする(ウィンドウ表示中はこちらを使う)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # ---- ③ 一定間隔で速度サマリーを表示 ----
            if frame_count % REPORT_INTERVAL == 0:
                avg_capture_ms = (sum(capture_times) / len(capture_times)) * 1000
                avg_process_ms = (sum(process_times) / len(process_times)) * 1000
                total_ms = avg_capture_ms + avg_process_ms
                fps_estimate = 1000 / total_ms if total_ms > 0 else float("inf")
                print(
                    f"[{frame_count}枚処理] 取得: {avg_capture_ms:.2f}ms, "
                    f"検出処理: {avg_process_ms:.2f}ms, "
                    f"合計: {total_ms:.2f}ms/frame (推定 {fps_estimate:.1f} fps), "
                    f"検出フレーム数: {detected_count}"
                )

    except KeyboardInterrupt:
        print("\n停止が要求されました。終了処理をします。")

    finally:
        session_elapsed = time.time() - session_start
        close_camera(cam_system, cam_device)
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

        print("\n--- セッションサマリー ---")
        print(f"総フレーム数: {frame_count}")
        print(f"検出できたフレーム数: {detected_count}")
        print(f"経過時間: {session_elapsed:.2f}秒")
        if frame_count > 0:
            print(f"平均fps(取得+検出処理込み): {frame_count / session_elapsed:.1f} fps")


if __name__ == "__main__":
    main()