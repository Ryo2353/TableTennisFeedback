# -*- coding: utf-8 -*-
"""
capture_dual.py の「録画エラーの原因調査用」版。
元の capture_dual.py はそのまま残し、このファイルを同じフォルダ(src/capture/)に置いて使う。

元のスクリプトからの変更点:
  1. 録画の上限枚数をコマンドラインで指定できる(--max-frames)
  2. 録画中、1秒ごとに状態を表示する
       - 取得枚数 / 書き込み済み枚数
       - 書き込み待ち(メモリにたまっている枚数)
       - コマ落ち数(block_id が飛んだ数)
       - ディスクの空き容量
  3. --no-save を付けると、画像を保存せずに「取得とコマ落ちの確認だけ」をする
       → USB(転送)の問題か、保存(ディスク・メモリ)の問題かを切り分けられる
  4. ディスクの空きが --min-free-gb を下回ったら、自動で録画を止める
  5. エラーが起きたとき、詳しい内容(トレースバック)を表示する
  6. 録画を止めたときに、カメラごとのまとめを表示する

使い方(src/capture/ フォルダで):
    python capture_dual_diag.py --max-frames 3000 --no-save   # 保存せず、USBだけ確認
    python capture_dual_diag.py --max-frames 1000             # 保存あり、1000枚
    python capture_dual_diag.py --max-frames 3000             # 保存あり、3000枚

キー操作は元と同じ('r'=録画開始/停止, 's'=スナップショット, 'q'=終了)。
"""

import argparse
import os
import queue
import shutil
import threading
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import pytelicam

EXPOSURE_TIME_US = 1000.0
GAIN_DB = 20.0
FRAME_SYNC_VALUE = "Bus"
DISPLAY_HEIGHT = 480
SAVE_FORMAT = ".bmp"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "data", "captures"))

# main() で上書きされる設定値
MAX_RECORD_FRAMES = 500
NO_SAVE = False


def free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


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
        self.capture_alive = True  # 取得スレッドがエラーで止まったら False

        # 連続録画用
        self.recording = False
        self.record_lock = threading.Lock()
        self.saved_count = 0          # 書き込みが終わった枚数
        self.write_fail_count = 0     # 書き込みに失敗した枚数
        self._queued_count = 0        # 録画中に取得した枚数
        self._last_saved_block_id = None
        # 調査用の記録
        self.drop_count = 0           # block_id が飛んだ数(=コマ落ち)
        self.max_queue = 0            # 書き込み待ちの最大枚数
        self.record_start_time = None

        self.save_queue = queue.Queue()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _writer_loop(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                self.save_queue.task_done()
                break
            arr, filename = item
            path = os.path.join(self.save_dir, filename)
            try:
                ok = cv2.imwrite(path, arr)
                if ok:
                    self.saved_count += 1
                else:
                    self.write_fail_count += 1
                    if self.write_fail_count <= 3:
                        print(f"[{self.label}] 書き込み失敗(imwriteがFalse): {path}")
            except Exception:
                self.write_fail_count += 1
                print(f"[{self.label}] 書き込み中に例外:")
                traceback.print_exc()
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
                            # コマ落ちの確認(block_id が2以上飛んだら、その間のコマを取りこぼしている)
                            if (self._last_saved_block_id is not None and block_id is not None
                                    and block_id - self._last_saved_block_id > 1):
                                self.drop_count += block_id - self._last_saved_block_id - 1

                            if not NO_SAVE:
                                filename = f"{self._queued_count:06d}_bid{block_id}_ts{ts}{SAVE_FORMAT}"
                                self.save_queue.put((arr_copy, filename))
                                self.max_queue = max(self.max_queue, self.save_queue.qsize())
                            self._last_saved_block_id = block_id
                            self._queued_count += 1
            except Exception:
                print(f"\n[{self.label}] 取得エラー(取得スレッドを終了します):")
                traceback.print_exc()
                print(f"[{self.label}] エラー時点: 録画中の取得枚数={self._queued_count}, "
                      f"書き込み済み={self.saved_count}, 書き込み待ち={self.save_queue.qsize()}")
                self.capture_alive = False
                with self.record_lock:
                    self.recording = False
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
            self.drop_count = 0
            self.max_queue = 0
            self.saved_count = 0
            self.write_fail_count = 0
            self.record_start_time = time.time()
        print(f"[{self.label}] 録画開始 -> {self.save_dir}" + ("(保存なしモード)" if NO_SAVE else ""))

    def stop_recording(self):
        with self.record_lock:
            self.recording = False
        self.print_summary()

    def print_summary(self):
        elapsed = time.time() - self.record_start_time if self.record_start_time else 0
        fps = self._queued_count / elapsed if elapsed > 0 else 0
        print(f"[{self.label}] --- まとめ ---")
        print(f"[{self.label}]   取得枚数: {self._queued_count}  録画時間: {elapsed:.1f}秒  実効fps: {fps:.1f}")
        print(f"[{self.label}]   コマ落ち(block_idの飛び): {self.drop_count}")
        if not NO_SAVE:
            print(f"[{self.label}]   書き込み済み: {self.saved_count}  書き込み失敗: {self.write_fail_count}  "
                  f"書き込み待ちの最大: {self.max_queue}  今の書き込み待ち: {self.save_queue.qsize()}")

    def status_line(self):
        s = f"{self.label}: 取得{self._queued_count} コマ落ち{self.drop_count}"
        if not NO_SAVE:
            s += f" 書込済{self.saved_count} 書込待ち{self.save_queue.qsize()}"
        if not self.capture_alive:
            s += " [取得停止]"
        return s

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
    global MAX_RECORD_FRAMES, NO_SAVE

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=500, help="1回の録画の上限枚数(1台あたり)")
    parser.add_argument("--no-save", action="store_true", help="画像を保存せず、取得とコマ落ちの確認だけ行う")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="ディスクの空きがこれを下回ったら録画を止める")
    args = parser.parse_args()
    MAX_RECORD_FRAMES = args.max_frames
    NO_SAVE = args.no_save

    session_dir = os.path.join(OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    print(f"保存先セッションフォルダ: {session_dir}")
    print(f"上限枚数: {MAX_RECORD_FRAMES}枚/台  保存: {'しない' if NO_SAVE else 'する'}  "
          f"ディスク空き: {free_gb(session_dir):.1f}GB")

    cam_system = pytelicam.get_camera_system()
    num = cam_system.get_num_of_cameras()
    print(f"接続カメラ台数: {num}")
    if num < 2:
        print("2台接続されていません。USB接続を確認してください。")
        cam_system.terminate()
        return

    cam_a = None
    cam_b = None
    try:
        cam_a = CameraWorker(cam_system, 0, "カメラA", os.path.join(session_dir, "camera_a"))
        cam_b = CameraWorker(cam_system, 1, "カメラB", os.path.join(session_dir, "camera_b"))

        start_barrier = threading.Barrier(2)
        cam_a.start(start_barrier)
        cam_b.start(start_barrier)

        print("\n操作方法:")
        print("  's' キー : 2台同時にスナップショットを1組保存")
        print("  'r' キー : 連続録画の開始/停止を切り替え")
        print("  'q' キー : 終了\n")

        snapshot_count = 0
        recording = False
        countdown_end_time = None
        RECORD_DELAY_SEC = 3.0
        last_status_time = 0.0

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

            cv2.imshow("Dual Capture DIAG (s:snapshot r:record q:quit)", combined)

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
                    recording = False
                    cam_a.stop_recording()
                    cam_b.stop_recording()
                elif countdown_end_time is not None:
                    countdown_end_time = None
                    print("録画開始をキャンセルしました")
                else:
                    countdown_end_time = time.time() + RECORD_DELAY_SEC
                    print(f"{RECORD_DELAY_SEC:.0f}秒後に録画を開始します...(もう一度 'r' でキャンセル)")

            if countdown_end_time is not None and time.time() >= countdown_end_time:
                countdown_end_time = None
                recording = True
                cam_a.start_recording()
                cam_b.start_recording()

            # 録画中は1秒ごとに状態を表示し、ディスクの空きを確認する
            now = time.time()
            if recording and now - last_status_time >= 1.0:
                last_status_time = now
                fgb = free_gb(session_dir)
                print(f"  {cam_a.status_line()} | {cam_b.status_line()} | 空き{fgb:.1f}GB")
                if not NO_SAVE and fgb < args.min_free_gb:
                    print(f"ディスクの空きが{args.min_free_gb}GBを下回ったため録画を止めます")
                    recording = False
                    cam_a.stop_recording()
                    cam_b.stop_recording()

            # 上限到達・取得エラーでどちらかが止まっていたら、両方揃えて停止する
            if recording and not (cam_a.recording and cam_b.recording):
                recording = False
                cam_a.stop_recording()
                cam_b.stop_recording()
                print("上限到達またはエラーにより両カメラの録画を停止しました")
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