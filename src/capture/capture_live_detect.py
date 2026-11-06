# -*- coding: utf-8 -*-
"""
撮りながらボールを検出し、画像は保存せずに「検出結果(位置)」だけを記録するツール。
(capture_dual.py + Detectball_frame.py を1つにしたもの)

仕組み:
    カメラごとに2つのスレッドが動く。
      取得スレッド : カメラから画像を受け取り、録画中ならキューに入れる
      検出スレッド : キューから1コマずつ取り出し、3フレーム差分でボールを検出する
    画像は保存しないので、ディスクやメモリがあふれない。
    検出が撮影に追いつかないとキューがたまる → 1秒ごとの表示で「処理待ち」を確認する。

出力(セッションフォルダ data/captures/YYYYMMDD_HHMMSS/ の中に、録画ごとに rec01, rec02, ... を作る):
    camera_a/detections.json   各コマの検出結果(Detectball_frame.py と同じ形 + delta)
    camera_b/detections.json
    camera_a/crops/, camera_b/crops/   検出したコマの周辺だけを切り出した小さな画像(確認用)
    session_info.json          録画開始時のbid(=delta 0)、コマ数、コマ落ち数など

    ※ delta = bid - 録画開始時のbid。2台の対応付けはこの delta が同じもの同士で行う
      (これまでの triangulate_trajectory.py と同じ考え方)。
      これまでは画像フォルダ全体から最小のbidを求めていたが、今は画像がないので、
      録画開始時のbidを session_info.json に記録し、detections.json にも delta を入れてある。

使い方(src/capture/ フォルダで):
    python capture_live_detect.py                 # 縮小1/2で検出(標準)
    python capture_live_detect.py --scale 1.0     # フル解像度で検出(元と同じ。重い)
    python capture_live_detect.py --no-crops      # 確認用の切り出し画像を保存しない

本番(撮影 → 着地点 → 投影を1回の操作で。プロジェクトのいちばん上のフォルダで):
    python src/capture/capture_live_detect.py --min-circ 0.1 --calib results/YYYYMMDD --projector results/YYYYMMDD/projector_transform.npz

    --calib を付けると、'r' で止めたときに自動で run_set.py(追跡 → 着地点 → ヒートマップ)を動かす。
    1セットは 'r' で開始してから止めるまでの1回。ヒートマップは投影の画面で Esc または q を押すと閉じ、
    カメラの画面に戻って次のセットを始められる。処理と投影の間は 'r' も 'q' も効かない。
    --projector を省くと投影せず、画像の保存だけをする。--calib がなければ今までと同じ動きをする。
    カメラの取得エラーで止まったときは、途中までのヒートマップを映さないよう、自動処理はしない。

キー操作:
    'r' : 録画(検出)の開始/停止
    'q' : 終了

本番・テストともに、ノートPCを電源につなぎ「最適なパフォーマンス」にすること。
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pytelicam

EXPOSURE_TIME_US = 1000.0
GAIN_DB = 20.0
FRAME_SYNC_VALUE = "Bus"
DISPLAY_HEIGHT = 360
DISPLAY_INTERVAL_SEC = 1 / 30   # 画面表示は30fps程度に間引く(検出の邪魔をしないため)

# ---- 検出条件(Detectball_frame.py と同じ。面積と明るさチェックの範囲は縮小率に合わせて自動で調整) ----
MIN_AREA = 100
MAX_AREA = 4000
MIN_CIRCULARITY = 0.3
MIN_BRIGHTNESS = 150
DIFF_THRESH = 50
KERNEL = np.ones((3, 3), np.uint8)

MAX_CANDIDATES = 5       # 1コマで残す候補の最大数(2台の組み合わせを後で選ぶため)
QUEUE_MAX = 400          # 処理待ちの上限(1台あたり。フル解像度で約600MB)。超えたら「処理落ち」として捨てる
CROP_HALF = 120          # 確認用の切り出し画像の半径(px、元画像の大きさで)。10/23 に 40→120(240×240)に拡大
MAX_RECORD_SEC = 180     # 安全のための録画時間の上限

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "data", "captures"))
_SRC_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
RUN_SET = os.path.join(_SRC_DIR, "run_set.py")
sys.path.insert(0, _SRC_DIR)
from run_set import check_calib


def detect_ball(prev, cur, nxt, scale):
    """
    3フレーム差分でボール候補を探し、(面積最大の候補, 候補数, 候補の一覧) を返す。
    prev, cur, nxt は縮小済みの画像。返す座標・面積は元画像の大きさに戻した値。
    候補の一覧は面積の大きい順に最大 MAX_CANDIDATES 個。手や体など、ボール以外が
    一番大きく写ったときでも、2台でつじつまの合う候補を後から選べるように全部残す。
    """
    diff = cv2.bitwise_and(cv2.absdiff(prev, cur), cv2.absdiff(cur, nxt))
    _, binary = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, KERNEL, iterations=2)
    binary = cv2.erode(binary, KERNEL, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    s2 = scale * scale
    r = max(2, int(round(5 * scale)))
    h, w = cur.shape[:2]
    best = None
    n = 0
    cands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA * s2 or area > MAX_AREA * s2:
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
        patch = cur[max(0, int(cy) - r):min(h, int(cy) + r), max(0, int(cx) - r):min(w, int(cx) + r)]
        if patch.size == 0 or patch.mean() < MIN_BRIGHTNESS:
            continue
        n += 1
        cands.append([cx / scale, cy / scale, area / s2])
        if best is None or area > best[2]:
            best = (cx / scale, cy / scale, area / s2, circularity)
    cands.sort(key=lambda c: -c[2])
    return best, n, cands[:MAX_CANDIDATES]


def refine_center(img, u, v, half=60):
    """
    3フレーム差分の重心は、遅い球では三日月の形になるため、ボールの中心からずれることがある。
    そこで、検出した位置のまわり(元の解像度の画像)で「明るい部分」を取り出し、
    その中でいちばん内側の点(距離変換の最大点)をボールの中心とする。
    距離変換を使うので、ボールに細い白線(台のライン)がくっついていても中心がぶれにくい。
    見つからなければ元の位置を返す。返り値: (u, v, 半径の目安)
    """
    h, w = img.shape[:2]
    x0, y0 = max(0, int(u) - half), max(0, int(v) - half)
    x1, y1 = min(w, int(u) + half), min(h, int(v) + half)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return u, v, 0.0
    mask = (cv2.GaussianBlur(roi, (3, 3), 0) >= MIN_BRIGHTNESS).astype(np.uint8)
    if mask.sum() == 0:
        return u, v, 0.0
    n, labels = cv2.connectedComponents(mask)
    # 検出位置にいちばん近い明るい塊を選ぶ
    cu, cv_ = int(u) - x0, int(v) - y0
    best_lab, best_d = None, None
    for lab in range(1, n):
        ys, xs = np.nonzero(labels == lab)
        dmin = np.min((xs - cu) ** 2 + (ys - cv_) ** 2)
        if best_d is None or dmin < best_d:
            best_lab, best_d = lab, dmin
    if best_lab is None or best_d > 30 ** 2:
        return u, v, 0.0
    blob = (labels == best_lab).astype(np.uint8)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    r = float(dist.max())
    if r < 3:
        return u, v, 0.0
    yy, xx = np.unravel_index(np.argmax(dist), dist.shape)
    # 最大点が平らに広がるとき(大きいボール)は、最大値の9割以上の点の平均にする
    ys, xs = np.nonzero(dist >= 0.9 * r)
    return float(x0 + xs.mean()), float(y0 + ys.mean()), r


class CameraWorker:
    def __init__(self, cam_system, index, label, cam_name, scale, save_crops):
        self.label = label
        self.cam_name = cam_name      # "camera_a" / "camera_b"(出力フォルダ名)
        self.out_dir = None           # 録画ごとに start_recording で決まる
        self.crop_dir = None
        self.scale = scale
        self.save_crops = save_crops

        self.device = cam_system.create_device_object(camera_index=index)
        self.device.open()
        print(f"[{label}] index={index} 情報: {self.device.get_information()}")
        try:
            self.device.genapi.set_float_value("ExposureTime", EXPOSURE_TIME_US)
            self.device.genapi.set_float_value("Gain", GAIN_DB)
        except Exception as e:
            print(f"[{label}] 露光/ゲイン設定に失敗: {e}")
        try:
            self.device.genapi.set_enum_str_value("FrameSynchronization", FRAME_SYNC_VALUE)
        except Exception as e:
            print(f"[{label}] FrameSynchronization の設定に失敗: {e}")
        self.device.cam_stream.open()

        # 表示用
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_bid = None
        self.last_det = None          # 表示用: 直近の検出 (u, v, 検出した時刻)
        self.stop_event = threading.Event()
        self.capture_alive = True

        # 録画(検出)の状態
        self.record_lock = threading.Lock()
        self.recording = False
        self.first_bid = None
        self.last_bid = None
        self.captured = 0             # 取得したコマ数
        self.drop_count = 0           # コマ落ち(bidの飛び)
        self.overflow_count = 0       # 処理落ち(キューがいっぱいで捨てたコマ)
        self.max_queue = 0
        self.record_start_time = None

        # 検出
        self.frame_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.processed = 0
        self.detections = []
        self.det_time_total = 0.0     # 検出にかかった時間の合計(秒)
        self.crop_queue = queue.Queue()

        self.capture_thread = None
        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.crop_thread = threading.Thread(target=self._crop_loop, daemon=True)
        self.detect_thread.start()
        self.crop_thread.start()

    # ---------------- 取得 ----------------
    def start(self, start_barrier):
        self.capture_thread = threading.Thread(target=self._capture_loop, args=(start_barrier,), daemon=True)
        self.capture_thread.start()

    def _capture_loop(self, start_barrier):
        start_barrier.wait()
        self.device.cam_stream.start()
        while not self.stop_event.is_set():
            try:
                image_data = self.device.cam_stream.get_next_image()
                arr = image_data.get_ndarray().copy()
                bid = getattr(image_data, "block_id", None)
                ts = getattr(image_data, "timestamp", None)
                image_data.release()
            except Exception as e:
                print(f"\n[{self.label}] 取得エラー(取得スレッドを終了します): {e}")
                self.capture_alive = False
                with self.record_lock:
                    self.recording = False
                break

            with self.lock:
                self.latest_frame = arr
                self.latest_bid = bid

            with self.record_lock:
                if not self.recording or bid == self.last_bid:
                    continue
                if self.first_bid is None:
                    self.first_bid = bid
                elif bid is not None and self.last_bid is not None and bid - self.last_bid > 1:
                    self.drop_count += bid - self.last_bid - 1
                self.last_bid = bid
                frame_num = self.captured
                self.captured += 1
                first_bid = self.first_bid

            try:
                self.frame_queue.put_nowait((frame_num, bid, ts, first_bid, arr))
                self.max_queue = max(self.max_queue, self.frame_queue.qsize())
            except queue.Full:
                self.overflow_count += 1

    # ---------------- 検出 ----------------
    def _detect_loop(self):
        buf = []   # 直近3コマ: (frame_num, bid, ts, first_bid, 元画像, 縮小画像)
        while True:
            item = self.frame_queue.get()
            if item is None:
                self.frame_queue.task_done()
                break
            if item == "reset":
                buf = []
                self.frame_queue.task_done()
                continue
            frame_num, bid, ts, first_bid, arr = item
            t0 = time.perf_counter()
            small = arr if self.scale == 1.0 else cv2.resize(
                arr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            buf.append((frame_num, bid, ts, first_bid, arr, small))
            if len(buf) > 3:
                buf.pop(0)
            if len(buf) == 3:
                f_num, f_bid, f_ts, f_first, f_arr, _ = buf[1]
                best, n, cands = detect_ball(buf[0][5], buf[1][5], buf[2][5], self.scale)
                if best is not None:
                    # 元の解像度の画像で、ボールの中心を求め直す(三日月の重心のずれを直す)
                    raw_u, raw_v = best[0], best[1]
                    u, v, _ = refine_center(f_arr, raw_u, raw_v)
                    _, _, area, circ = best
                    for c in cands:
                        c[0], c[1], _ = refine_center(f_arr, c[0], c[1])
                    det = {
                        "frame_num": f_num,
                        "bid": f_bid,
                        "ts": f_ts,
                        "delta": (f_bid - f_first) if (f_bid is not None and f_first is not None) else None,
                        "u": float(u),
                        "v": float(v),
                        "u_raw": float(raw_u),   # 差分の重心(求め直す前の位置)
                        "v_raw": float(raw_v),
                        "area": float(area),
                        "circularity": float(circ),
                        "num_candidates": n,
                        "candidates": [[float(c[0]), float(c[1]), float(c[2])] for c in cands],
                    }
                    self.detections.append(det)
                    with self.lock:
                        self.last_det = (u, v, time.time())
                    if self.save_crops:
                        self.crop_queue.put((f_arr, u, v, f_num, f_bid, self.crop_dir))
            self.det_time_total += time.perf_counter() - t0
            self.processed += 1
            self.frame_queue.task_done()

    def _crop_loop(self):
        while True:
            item = self.crop_queue.get()
            if item is None:
                break
            arr, u, v, f_num, f_bid, crop_dir = item
            h, w = arr.shape[:2]
            x0, y0 = max(0, int(u) - CROP_HALF), max(0, int(v) - CROP_HALF)
            x1, y1 = min(w, int(u) + CROP_HALF), min(h, int(v) + CROP_HALF)
            name = f"{f_num:06d}_bid{f_bid}_u{int(u)}_v{int(v)}.png"
            cv2.imwrite(os.path.join(crop_dir, name), arr[y0:y1, x0:x1])

    # ---------------- 録画の開始・停止 ----------------
    def start_recording(self, rec_dir):
        self.out_dir = os.path.join(rec_dir, self.cam_name)
        self.crop_dir = os.path.join(self.out_dir, "crops")
        os.makedirs(self.out_dir, exist_ok=True)
        if self.save_crops:
            os.makedirs(self.crop_dir, exist_ok=True)
        self.frame_queue.put("reset")   # 前の録画の残りと混ざらないように
        with self.record_lock:
            self.recording = True
            self.first_bid = None
            self.last_bid = None
            self.captured = 0
            self.drop_count = 0
            self.overflow_count = 0
            self.max_queue = 0
            self.record_start_time = time.time()
        self.processed = 0
        self.detections = []
        self.det_time_total = 0.0
        print(f"[{self.label}] 録画(検出)開始  縮小率 {self.scale}")

    def stop_recording(self):
        with self.record_lock:
            self.recording = False

    def wait_processed(self, timeout=30):
        """キューに残ったコマの検出が終わるまで待つ"""
        end = time.time() + timeout
        while self.frame_queue.unfinished_tasks > 0 and time.time() < end:
            time.sleep(0.05)

    def summary(self):
        elapsed = time.time() - self.record_start_time if self.record_start_time else 0
        avg_ms = self.det_time_total / self.processed * 1000 if self.processed else 0
        return {
            "label": self.label,
            "first_bid": self.first_bid,
            "last_bid": self.last_bid,
            "captured": self.captured,
            "processed": self.processed,
            "detected_frames": len(self.detections),
            "drop_count": self.drop_count,
            "overflow_count": self.overflow_count,
            "max_queue": self.max_queue,
            "record_sec": round(elapsed, 2),
            "fps": round(self.captured / elapsed, 1) if elapsed > 0 else 0,
            "avg_detect_ms": round(avg_ms, 3),
        }

    def save_results(self):
        path = os.path.join(self.out_dir, "detections.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.detections, f, ensure_ascii=False, indent=2)
        return path

    def status_line(self):
        s = (f"{self.label}: 取得{self.captured} 処理{self.processed} 処理待ち{self.frame_queue.qsize()} "
             f"検出{len(self.detections)} コマ落ち{self.drop_count}")
        if self.overflow_count:
            s += f" 処理落ち{self.overflow_count}"
        if not self.capture_alive:
            s += " [取得停止]"
        return s

    def get_display(self):
        with self.lock:
            return self.latest_frame, self.latest_bid, self.last_det

    def close(self):
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2)
        try:
            self.device.cam_stream.stop()
            self.device.cam_stream.close()
        except Exception:
            pass
        self.device.close()
        self.frame_queue.put(None)
        self.detect_thread.join(timeout=5)
        self.crop_queue.put(None)
        self.crop_thread.join(timeout=10)


def make_display(frame, bid, last_det, label, recording):
    if frame is None:
        img = np.zeros((DISPLAY_HEIGHT, 480, 3), dtype=np.uint8)
        cv2.putText(img, f"{label}: waiting...", (20, DISPLAY_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img
    h, w = frame.shape[:2]
    s = DISPLAY_HEIGHT / h
    small = cv2.resize(frame, (int(w * s), DISPLAY_HEIGHT))
    img = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR) if small.ndim == 2 else small
    # 直近0.5秒以内の検出位置に丸を描く
    if last_det is not None and time.time() - last_det[2] < 0.5:
        cv2.circle(img, (int(last_det[0] * s), int(last_det[1] * s)), 12, (0, 0, 255), 2)
    color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(img, f"{label} bid={bid}" + (" [REC]" if recording else ""), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def run_after_stop(rec_dir, args):
    """止めた後の処理(run_set.py)を動かし、終わるまで待つ。カメラの画面が応答なしにならないよう、
    待つ間も画面の処理は回す(キーは無視する)。失敗しても撮影は続けられるようにする。"""
    cmd = [sys.executable, RUN_SET, rec_dir, "--calib", args.calib]
    if args.projector:
        cmd += ["--projector", args.projector]
    if args.table:
        cmd += ["--table", args.table]
    if args.points:
        cmd.append("--points")
    if args.outline:
        cmd.append("--outline")
    print("自動で処理します(終わるまで 'r' と 'q' は効きません)")
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        cv2.waitKey(50)
    if proc.returncode != 0:
        print("[警告] 自動の処理が途中で止まりました。検出結果は残っています。あとで次のコマンドで動かせます:")
        print("  python src/run_set.py " + " ".join(cmd[2:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.5, help="検出時の縮小率(0.5=半分、1.0=縮小なし)")
    parser.add_argument("--no-crops", action="store_true", help="確認用の切り出し画像を保存しない")
    parser.add_argument("--crop-half", type=int, default=CROP_HALF,
                        help=f"切り出し画像の半径(画素。既定 {CROP_HALF} → {2 * CROP_HALF}×{2 * CROP_HALF})")
    parser.add_argument("--min-circ", type=float, default=MIN_CIRCULARITY,
                        help=f"円形度の下限(既定 {MIN_CIRCULARITY}。10/23 hitball6 で 0.1 にすると途切れが減った)")
    parser.add_argument("--calib", default=None,
                        help="stereo_result.npz と table_transform.npz のフォルダ。指定すると止めたときに自動で処理する")
    parser.add_argument("--projector", default=None, help="projector_transform.npz。省略すると投影しない")
    parser.add_argument("--table", default=None, help="台の名前(src/table_config.py)。省略すると今の設定")
    parser.add_argument("--points", action="store_true", help="着地点そのものも重ねて映す")
    parser.add_argument("--outline", action="store_true", help="計測範囲の枠を映す")
    args = parser.parse_args()
    if args.calib:
        check_calib(args.calib, args.projector, args.table)   # 撮り始めてから気づかないよう、最初に確かめる
    elif args.projector or args.table or args.points or args.outline:
        parser.error("--projector などは --calib と一緒に指定してください")

    session_dir = os.path.join(OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    print(f"保存先セッションフォルダ: {session_dir}")
    globals()["MIN_CIRCULARITY"] = args.min_circ
    globals()["CROP_HALF"] = args.crop_half
    print(f"検出の縮小率: {args.scale}  円形度の下限: {args.min_circ}  切り出しの大きさ: {2 * args.crop_half}画素  切り出し画像: {'保存しない' if args.no_crops else '保存する'}")

    cam_system = pytelicam.get_camera_system()
    num = cam_system.get_num_of_cameras()
    print(f"接続カメラ台数: {num}")
    if num < 2:
        print("2台接続されていません。USB接続を確認してください。")
        cam_system.terminate()
        return

    cam_a = cam_b = None
    recording = False
    try:
        cam_a = CameraWorker(cam_system, 0, "カメラA", "camera_a", args.scale, not args.no_crops)
        cam_b = CameraWorker(cam_system, 1, "カメラB", "camera_b", args.scale, not args.no_crops)
        barrier = threading.Barrier(2)
        cam_a.start(barrier)
        cam_b.start(barrier)

        print("\n操作方法:  'r' : 録画(検出)の開始/停止   'q' : 終了\n")
        last_status = 0.0
        last_display = 0.0
        countdown_end = None
        record_count = 0
        rec_dir = None

        def finish_recording():
            cam_a.stop_recording()
            cam_b.stop_recording()
            print("録画を停止しました。残りの検出を待っています...")
            cam_a.wait_processed()
            cam_b.wait_processed()
            sa, sb = cam_a.summary(), cam_b.summary()
            pa, pb = cam_a.save_results(), cam_b.save_results()
            info = {"record_index": record_count, "scale": args.scale, "min_circ": args.min_circ, "camera_a": sa, "camera_b": sb}
            with open(os.path.join(rec_dir, "session_info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            print("--- まとめ ---")
            for s in (sa, sb):
                print(f"[{s['label']}] 取得{s['captured']}コマ  {s['record_sec']}秒  {s['fps']}fps  "
                      f"検出{s['detected_frames']}コマ  コマ落ち{s['drop_count']}  処理落ち{s['overflow_count']}  "
                      f"処理待ちの最大{s['max_queue']}  検出の平均{s['avg_detect_ms']}ms/コマ")
            print(f"検出結果: {pa}\n          {pb}\n")
            if args.calib:
                if cam_a.capture_alive and cam_b.capture_alive:
                    run_after_stop(rec_dir, args)
                else:
                    print("[警告] カメラの取得が止まったので、自動の処理はしません。")

        while True:
            now = time.time()
            if now - last_display >= DISPLAY_INTERVAL_SEC:
                last_display = now
                fa, ba, da = cam_a.get_display()
                fb, bb, db = cam_b.get_display()
                combined = cv2.hconcat([make_display(fa, ba, da, "Camera A", recording),
                                        make_display(fb, bb, db, "Camera B", recording)])
                if countdown_end is not None and countdown_end - now > 0:
                    cv2.putText(combined, f"REC starts in {countdown_end - now:.1f}s", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                cv2.imshow("Live Detect (r:record q:quit)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if recording:
                    recording = False
                    finish_recording()
                elif countdown_end is not None:
                    countdown_end = None
                    print("録画開始をキャンセルしました")
                else:
                    countdown_end = now + 3.0
                    print("3秒後に録画(検出)を開始します...(もう一度 'r' でキャンセル)")

            if countdown_end is not None and now >= countdown_end:
                countdown_end = None
                record_count += 1
                rec_dir = os.path.join(session_dir, f"rec{record_count:02d}")
                recording = True
                cam_a.start_recording(rec_dir)
                cam_b.start_recording(rec_dir)
                print(f"録画{record_count}回目 -> {rec_dir}")

            if recording and now - last_status >= 1.0:
                last_status = now
                print(f"  {cam_a.status_line()} | {cam_b.status_line()}")

            if recording and (not (cam_a.recording and cam_b.recording)
                              or now - cam_a.record_start_time > MAX_RECORD_SEC):
                recording = False
                print("エラーまたは時間の上限により録画を停止します")
                finish_recording()

        if recording:
            recording = False
            finish_recording()
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