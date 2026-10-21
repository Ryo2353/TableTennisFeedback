# -*- coding: utf-8 -*-
"""
2台のカメラを同時に確実に制御しながら、画像を保存するツール。

やっていること:
- 2台のカメラを別スレッドで取得し続け、常に「最新の1枚」を保持する(dual_camera_viewer.pyと同じ仕組み)
- ライブ映像を見ながら、キー操作で以下ができる:
    's' キー : その瞬間の2台の画像を1組だけ保存する(キャリブレーション用ロッド撮影に)
    'r' キー : 連続録画の開始/停止を切り替える(ボール打撃の撮影に)
    'q' キー : 終了

保存先:
    <このファイルがあるフォルダ>/../../data/captures/YYYYMMDD_HHMMSS/camera_a/ と camera_b/ に、
    それぞれ block_id と timestamp をファイル名に含めて保存される。
    (2台のファイル名からblock_idを見れば、後で正しく対応するフレーム同士を突き合わせられる)
    保存先はこのスクリプト自身の場所を基準に決めているので、どのフォルダから実行しても
    同じ場所(プロジェクトの data/captures/)に保存される。
    src/capture/ 以外の場所にこのファイルを置く場合は OUTPUT_ROOT を書き換えること。

使い方:
    python capture_dual.py
"""

import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pytelicam

EXPOSURE_TIME_US = 1000.0
GAIN_DB = 20.0
FRAME_SYNC_VALUE = "Bus"
DISPLAY_HEIGHT = 480

MAX_RECORD_FRAMES = 500  # 1回の録画で保存する最大枚数(安全のための上限)
SAVE_FORMAT = ".bmp"  # 圧縮なしで書き込みを高速化(容量は増える)

# このファイル(src/capture/capture_dual.py)から見て ../../data/captures を保存先にする。
# 実行時のカレントフォルダに依存しないよう、__file__基準の絶対パスで解決する。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "data", "captures"))


class CameraWorker:
    def __init__(self, cam_system, index, label, save_dir):
        self.label = label
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = cam_system.create_device_object(camera_index=index)
        self.device.open()

        info = self.device.get_information()
        print(f"[{label}] index={index} 情報: {info}")

        try:
            self.device.genapi.set_float_value("ExposureTime", EXPOSURE_TIME_US)
            self.device.genapi.set_float_value("Gain", GAIN_DB)
        except Exception as e:
            print(f"[{label}] 露光/ゲイン設定に失敗: {e}")

        # NOTE: FrameSynchronization=Bus の設定はカメラに対して試みているが、
        # 実機検証(sync_test.py)では10秒間に約78msのドリフトが確認されており、
        # このBus設定だけに頼って正確な同時性を保証することはできない。
        # そのため実際のフレーム対応付けは、この設定が効いているかに関わらず、
        # ソフトウェアタイムスタンプ(block_id基準のオフライン突き合わせ、
        # offline_sync_check.py参照)で行う方針としている。
        try:
            self.device.genapi.set_enum_str_value("FrameSynchronization", FRAME_SYNC_VALUE)
            print(f"[{label}] FrameSynchronization を {FRAME_SYNC_VALUE} に設定しました")
        except Exception as e:
            print(f"[{label}] FrameSynchronization の設定に失敗: {e}")

        self.device.cam_stream.open()

        self.latest_frame = None
        self.latest_block_id = None
        self.latest_timestamp = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        # 連続録画用
        self.recording = False
        self.record_lock = threading.Lock()
        self.saved_count = 0
        self._queued_count = 0
        self._last_saved_block_id = None

        # 保存はキュー経由で別スレッドが行う(取得ループを保存待ちでブロックしないため)
        self.save_queue = queue.Queue()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _writer_loop(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            arr, filename = item
            path = os.path.join(self.save_dir, filename)
            cv2.imwrite(path, arr)
            self.save_queue.task_done()

    def start(self, start_barrier):
        self.thread = threading.Thread(target=self._capture_loop, args=(start_barrier,), daemon=True)
        self.thread.start()

    def _capture_loop(self, start_barrier):
        start_barrier.wait()
        self.device.cam_stream.start()

        while not self.stop_event.is_set():
            try:
                image_data = self.device.cam_stream.get_next_image()
                arr = image_data.get_ndarray()
                block_id = getattr(image_data, "block_id", None)
                ts = getattr(image_data, "timestamp", None)
                # copy推奨(releaseするとバッファが再利用されるため)
                arr_copy = arr.copy()
                image_data.release()

                with self.lock:
                    self.latest_frame = arr_copy
                    self.latest_block_id = block_id
                    self.latest_timestamp = ts

                with self.record_lock:
                    if self.recording and block_id != self._last_saved_block_id:
                        if self._queued_count >= MAX_RECORD_FRAMES:
                            self.recording = False
                            print(f"[{self.label}] 最大枚数({MAX_RECORD_FRAMES}枚)に達したため録画を自動停止しました")
                        else:
                            filename = f"{self._queued_count:06d}_bid{block_id}_ts{ts}{SAVE_FORMAT}"
                            self.save_queue.put((arr_copy, filename))
                            self._last_saved_block_id = block_id
                            self._queued_count += 1
            except Exception as e:
                print(f"[{self.label}] 取得エラー: {e}")
                break

    def save_snapshot(self, tag=""):
        with self.lock:
            frame = self.latest_frame
            block_id = self.latest_block_id
            ts = self.latest_timestamp
        if frame is None:
            print(f"[{self.label}] まだ画像がありません")
            return
        filename = f"snapshot_{tag}_bid{block_id}_ts{ts}.png"
        path = os.path.join(self.save_dir, filename)
        cv2.imwrite(path, frame)
        print(f"[{self.label}] スナップショット保存: {path}")

    def start_recording(self):
        with self.record_lock:
            self.recording = True
            self._queued_count = 0
            self._last_saved_block_id = None
        print(f"[{self.label}] 録画開始 -> {self.save_dir}")

    def stop_recording(self):
        with self.record_lock:
            self.recording = False
        print(f"[{self.label}] 録画停止(取得枚数: {self._queued_count}、書き込みはバックグラウンドで継続中)")

    def get_latest(self):
        with self.lock:
            return self.latest_frame, self.latest_block_id, self.latest_timestamp

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        try:
            self.device.cam_stream.stop()
            self.device.cam_stream.close()
        except Exception:
            pass
        self.device.close()

        # 保存キューが残っていれば書き終わるまで待つ
        remaining = self.save_queue.qsize()
        if remaining > 0:
            print(f"[{self.label}] 残り{remaining}枚を書き込み中...")
        self.save_queue.join()
        self.save_queue.put(None)
        self.writer_thread.join(timeout=2)


def prepare_for_display(frame, block_id, timestamp, label, recording):
    if frame is None:
        img = np.zeros((DISPLAY_HEIGHT, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"{label}: waiting...", (20, DISPLAY_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img

    if frame.ndim == 2:
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        img = frame.copy()

    h, w = img.shape[:2]
    scale = DISPLAY_HEIGHT / h
    img = cv2.resize(img, (int(w * scale), DISPLAY_HEIGHT))

    color = (0, 0, 255) if recording else (0, 255, 0)
    text1 = f"{label}  block_id={block_id}" + ("  [REC]" if recording else "")
    text2 = f"ts={timestamp}"
    cv2.putText(img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return img


def main():
    session_dir = os.path.join(OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    print(f"保存先セッションフォルダ: {session_dir}")

    cam_system = pytelicam.get_camera_system()
    num = cam_system.get_num_of_cameras()
    print(f"接続カメラ台数: {num}")
    if num < 2:
        print("2台接続されていません。USB接続を確認してください。")
        cam_system.terminate()
        return

    # cam_a / cam_bどちらかの初期化中に例外が起きても、既に開いた方を
    # 必ず閉じるようにtry/finallyで囲む(片方だけ開いたまま終了すると、
    # 次回実行時に「AlreadyOpened」エラーが連鎖する原因になるため)
    cam_a = None
    cam_b = None
    try:
        cam_a = CameraWorker(cam_system, 0, "カメラA", os.path.join(session_dir, "camera_a"))
        cam_b = CameraWorker(cam_system, 1, "カメラB", os.path.join(session_dir, "camera_b"))

        start_barrier = threading.Barrier(2)
        cam_a.start(start_barrier)
        cam_b.start(start_barrier)

        print("\n操作方法:")
        print("  's' キー : 2台同時にスナップショットを1組保存(キャリブレーション用)")
        print("  'r' キー : 連続録画の開始/停止を切り替え(ボール撮影用)")
        print("  'q' キー : 終了\n")

        snapshot_count = 0
        recording = False
        countdown_end_time = None
        RECORD_DELAY_SEC = 3.0

        while True:
            frame_a, bid_a, ts_a = cam_a.get_latest()
            frame_b, bid_b, ts_b = cam_b.get_latest()

            img_a = prepare_for_display(frame_a, bid_a, ts_a, "Camera A", recording)
            img_b = prepare_for_display(frame_b, bid_b, ts_b, "Camera B", recording)
            combined = cv2.hconcat([img_a, img_b])

            if countdown_end_time is not None:
                remaining = countdown_end_time - time.time()
                if remaining > 0:
                    cv2.putText(combined, f"REC starts in {remaining:.1f}s", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

            cv2.imshow("Dual Capture (s:snapshot r:record q:quit)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snapshot_count += 1
                tag = f"{snapshot_count:05d}"
                cam_a.save_snapshot(tag)
                cam_b.save_snapshot(tag)
            elif key == ord("r"):
                if recording:
                    # 録画中は即座に停止する
                    recording = False
                    cam_a.stop_recording()
                    cam_b.stop_recording()
                elif countdown_end_time is not None:
                    # カウントダウン中にもう一度押したらキャンセル
                    countdown_end_time = None
                    print("録画開始をキャンセルしました")
                else:
                    countdown_end_time = time.time() + RECORD_DELAY_SEC
                    print(f"{RECORD_DELAY_SEC:.0f}秒後に録画を開始します...(もう一度 'r' でキャンセル)")

            # カウントダウンが終わったら実際に録画開始する
            if countdown_end_time is not None and time.time() >= countdown_end_time:
                countdown_end_time = None
                recording = True
                cam_a.start_recording()
                cam_b.start_recording()

            # 上限枚数到達によりワーカー側が自動停止していたら、両方揃えて停止する
            if recording and not (cam_a.recording and cam_b.recording):
                recording = False
                cam_a.stop_recording()
                cam_b.stop_recording()
                print("上限到達により両カメラの録画を停止しました")
    finally:
        if cam_a is not None:
            cam_a.close()
        if cam_b is not None:
            cam_b.close()
        cam_system.terminate()
        cv2.destroyAllWindows()
        print(f"\n保存先: {session_dir}")


if __name__ == "__main__":
    main()