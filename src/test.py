"""
2台のカメラ(真上カメラ・横カメラ)が同期しているかを検証するスクリプト。

やっていること:
1. 2台のカメラをそれぞれ開き、露光・ゲインを揃え、FrameSynchronizationをBus(バス同期)に設定
2. 2つのスレッドで、それぞれのカメラから同時にフレームを取得し続け、
   フレームが届いた瞬間のPC側の時刻(タイムスタンプ)を記録する
3. 記録が終わったら、「カメラAのn枚目」と「カメラBのn枚目」が届いた時刻の差を計算し、
   その差が時間とともに安定しているか(同期できている)、
   それともじわじわ広がっていくか(同期できていない)を確認する

使い方:
    pip install pytelicam numpy
    python sync_test.py

前提:
    2台のカメラが同じUSB3.0ハブに接続されていること。
    FrameSynchronizationの正確な値名("Bus"など)は、環境によって
    大文字/小文字が違う可能性があるので、エラーが出た場合は
    d.genapi.get_available_enum_entry_names("FrameSynchronization")
    で実際の値を確認してください。
"""

import threading
import time

import numpy as np
import pytelicam

# ============================================================
# 設定
# ============================================================
EXPOSURE_TIME_US = 1000.0
GAIN_DB = 20.0
FRAME_SYNC_VALUE = "Bus"   # バス同期。環境によって "bus" など大文字小文字が違う可能性あり

CAPTURE_SECONDS = 10  # 何秒間、記録し続けるか


def setup_camera(cam_system, index, label):
    """カメラを開き、露光・ゲイン・バス同期を設定する"""
    device = cam_system.create_device_object(camera_index=index)
    device.open()

    info = device.get_information()
    print(f"[{label}] index={index} 情報: {info}")

    try:
        device.genapi.set_float_value("ExposureTime", EXPOSURE_TIME_US)
        device.genapi.set_float_value("Gain", GAIN_DB)
    except Exception as e:
        print(f"[{label}] 露光/ゲイン設定に失敗: {e}")

    try:
        device.genapi.set_enum_str_value("FrameSynchronization", FRAME_SYNC_VALUE)
        print(f"[{label}] FrameSynchronization を {FRAME_SYNC_VALUE} に設定しました")
    except Exception as e:
        print(f"[{label}] FrameSynchronization の設定に失敗: {e}")
        try:
            options = device.genapi.get_available_enum_entry_names("FrameSynchronization")
            print(f"[{label}] 実際に使える値: {options}")
        except Exception:
            pass

    device.cam_stream.open()
    # NOTE: start()はここでは呼ばない。2台同時にstart()するため、
    # 呼び出し側でBarrierを使って揃えてからstart()する。
    return device


def capture_loop(device, label, timestamps, stop_event, start_barrier):
    """別スレッドで、2台同時にstart()してから、フレームが届くたびに時刻を記録し続ける"""
    # 両スレッドがここに到達するまで待ち、揃った瞬間にstart()する。
    # これにより「片方だけ先に貯め始めている」という頭出しのズレを解消する。
    start_barrier.wait()
    device.cam_stream.start()

    while not stop_event.is_set():
        try:
            image_data = device.cam_stream.get_next_image()
            t = time.time()

            # カメラ内部のタイムスタンプ/フレーム番号が取得できるか試す
            # (属性名は環境依存のため候補を複数試す。取れなければNoneのまま)
            hw_ts = None
            hw_frame_id = None
            for attr in ("get_timestamp", "timestamp"):
                if hasattr(image_data, attr):
                    try:
                        val = getattr(image_data, attr)
                        hw_ts = val() if callable(val) else val
                        break
                    except Exception:
                        pass
            # block_id がフレームの連番(カメラ内部カウンタ)に相当する
            if hasattr(image_data, "block_id"):
                try:
                    val = image_data.block_id
                    hw_frame_id = val() if callable(val) else val
                except Exception:
                    pass

            image_data.release()
            timestamps.append((t, hw_ts, hw_frame_id))
        except Exception as e:
            print(f"[{label}] 取得エラー: {e}")
            break


def main():
    cam_system = pytelicam.get_camera_system()
    num = cam_system.get_num_of_cameras()
    print(f"接続カメラ台数: {num}")
    if num < 2:
        print("2台接続されていません。USB接続を確認してください。")
        return

    device_a = setup_camera(cam_system, 0, "カメラA(index=0)")
    device_b = setup_camera(cam_system, 1, "カメラB(index=1)")

    timestamps_a = []
    timestamps_b = []
    stop_event = threading.Event()
    start_barrier = threading.Barrier(2)  # 2台が揃った瞬間にstart()させる

    thread_a = threading.Thread(target=capture_loop, args=(device_a, "カメラA", timestamps_a, stop_event, start_barrier))
    thread_b = threading.Thread(target=capture_loop, args=(device_b, "カメラB", timestamps_b, stop_event, start_barrier))

    print(f"\n{CAPTURE_SECONDS}秒間、2台同時に取得します(start()タイミングを揃え済み)...")
    thread_a.start()
    thread_b.start()

    time.sleep(CAPTURE_SECONDS)
    stop_event.set()
    thread_a.join()
    thread_b.join()

    # 2台同時実行中のエラーカウントを確認(枚数ズレの原因切り分け用)
    acquired_a = device_a.cam_stream.get_image_acquired_count()
    error_a = device_a.cam_stream.get_image_error_count()
    acquired_b = device_b.cam_stream.get_image_acquired_count()
    error_b = device_b.cam_stream.get_image_error_count()
    print(f"\nカメラA: 取得数(カメラ内部カウント)={acquired_a}, エラー数={error_a}")
    print(f"カメラB: 取得数(カメラ内部カウント)={acquired_b}, エラー数={error_b}")

    device_a.cam_stream.stop()
    device_a.cam_stream.close()
    device_a.close()
    device_b.cam_stream.stop()
    device_b.cam_stream.close()
    device_b.close()
    cam_system.terminate()

    print(f"\nカメラA 取得枚数: {len(timestamps_a)}")
    print(f"カメラB 取得枚数: {len(timestamps_b)}")

    # ============================================================
    # 分析: n枚目同士の到着時刻の差を計算
    # ============================================================
    n = min(len(timestamps_a), len(timestamps_b))
    if n < 10:
        print("記録できたフレーム数が少なすぎます。撮影時間を延ばすか、接続を確認してください。")
        return

    # ============================================================
    # 分析: frame_id を使って、本当に対応するフレーム同士をペアリングする
    # ============================================================
    frame_ids_a = [x[2] for x in timestamps_a]
    frame_ids_b = [x[2] for x in timestamps_b]
    has_frame_id = (
        len(frame_ids_a) > 0 and len(frame_ids_b) > 0
        and all(v is not None for v in frame_ids_a[:10])
        and all(v is not None for v in frame_ids_b[:10])
    )

    if has_frame_id:
        # 各カメラの最初のframe_idを基準に「経過フレーム数(delta)」を計算する。
        # これにより、途中でコマ落ちしてリストの長さがズレていても、
        # 同じdelta(=同じタイミングのはずのフレーム)同士を正しく対応付けられる。
        base_a = frame_ids_a[0]
        base_b = frame_ids_b[0]
        delta_to_index_a = {fid - base_a: i for i, fid in enumerate(frame_ids_a)}
        delta_to_index_b = {fid - base_b: i for i, fid in enumerate(frame_ids_b)}

        common_deltas = sorted(set(delta_to_index_a.keys()) & set(delta_to_index_b.keys()))
        print(f"\n[frame_id基準のペアリング] 共通して対応付けられたフレーム数: {len(common_deltas)}")
        print(f"(単純なn枚目比較では{n}枚でしたが、frame_id基準では{len(common_deltas)}枚が正しく対応付けられました)")

        if len(common_deltas) >= 10:
            idx_a = [delta_to_index_a[d] for d in common_deltas]
            idx_b = [delta_to_index_b[d] for d in common_deltas]

            host_a2 = np.array([timestamps_a[i][0] for i in idx_a])
            host_b2 = np.array([timestamps_b[i][0] for i in idx_b])
            diffs_ms2 = (host_a2 - host_b2) * 1000

            print("\n=== [frame_id基準] フレーム到着時刻の差(A - B) ===")
            print(f"平均: {diffs_ms2.mean():.3f} ms")
            print(f"標準偏差: {diffs_ms2.std():.3f} ms")

            half2 = len(common_deltas) // 2
            drift2 = diffs_ms2[half2:].mean() - diffs_ms2[:half2].mean()
            print(f"前半平均: {diffs_ms2[:half2].mean():.3f} ms / 後半平均: {diffs_ms2[half2:].mean():.3f} ms")
            print(f"ドリフト(変化量): {drift2:.3f} ms")

            hw_ts_a_list2 = [timestamps_a[i][1] for i in idx_a]
            hw_ts_b_list2 = [timestamps_b[i][1] for i in idx_b]
            if all(v is not None for v in hw_ts_a_list2) and all(v is not None for v in hw_ts_b_list2):
                hw_a2 = np.array(hw_ts_a_list2, dtype=np.int64)
                hw_b2 = np.array(hw_ts_b_list2, dtype=np.int64)
                hw_diff2 = hw_a2 - hw_b2
                print("\n=== [frame_id基準] カメラ内部タイムスタンプの差(A - B) ===")
                print(f"標準偏差: {hw_diff2.std():.1f}")
                hw_drift2 = hw_diff2[half2:].mean() - hw_diff2[:half2].mean()
                print(f"ドリフト(変化量): {hw_drift2:.1f}  (ns想定なら {hw_drift2/1e6:.3f} ms)")
        else:
            print("共通フレーム数が少なすぎて分析できません。")
    else:
        print("\n[参考] frame_idが取得できなかったため、frame_id基準の分析はスキップします。")

    print("\n" + "=" * 60)
    print("[診断] 各カメラ単体でのコマ落ち検出(block_idの連番チェック)")
    print("=" * 60)
    for label, fids in (("カメラA", frame_ids_a), ("カメラB", frame_ids_b)):
        if not fids or fids[0] is None:
            print(f"{label}: block_id取得不可のためスキップ")
            continue
        gaps = []
        for i in range(1, len(fids)):
            step = fids[i] - fids[i - 1]
            if step != 1:
                gaps.append((i, fids[i - 1], fids[i], step))
        total_missed = sum(g[3] - 1 for g in gaps)
        print(f"{label}: 記録フレーム数={len(fids)}, 連番の飛び={len(gaps)}箇所, 推定コマ落ち枚数={total_missed}")
        for i, (idx, prev, cur, step) in enumerate(gaps[:5]):
            print(f"    位置{idx}付近: block_id {prev} -> {cur} (計{step-1}枚欠落)")
        if len(gaps) > 5:
            print(f"    ...ほか{len(gaps)-5}箇所")

    print("\n" + "=" * 60)
    print("以下は従来通り、単純に「n枚目」同士を比較した結果です(参考)")
    print("=" * 60)

    host_times_a = np.array([x[0] for x in timestamps_a[:n]])
    host_times_b = np.array([x[0] for x in timestamps_b[:n]])
    diffs = host_times_a - host_times_b
    diffs_ms = diffs * 1000

    # カメラ内部タイムスタンプが取得できていたか確認
    hw_ts_a_list = [x[1] for x in timestamps_a[:n]]
    hw_ts_b_list = [x[1] for x in timestamps_b[:n]]
    has_hw_ts = all(v is not None for v in hw_ts_a_list) and all(v is not None for v in hw_ts_b_list)

    if has_hw_ts:
        hw_ts_a = np.array(hw_ts_a_list, dtype=np.int64)
        hw_ts_b = np.array(hw_ts_b_list, dtype=np.int64)

        print(f"\n[参考] カメラ内部タイムスタンプ(先頭値): A={hw_ts_a[0]}, B={hw_ts_b[0]}")

        # 単位はカメラの実装依存(ns想定)。まずは生の差分を確認する。
        hw_diff = hw_ts_a - hw_ts_b
        hw_diff_abs = np.abs(hw_diff)

        print("\n=== カメラ内部タイムスタンプの差分(A - B) ===")
        print(f"平均: {hw_diff.mean():.1f}  (単位不明。ns想定なら /1e6 でms換算)")
        print(f"平均(|差|): {hw_diff_abs.mean():.1f}")
        print(f"標準偏差: {hw_diff.std():.1f}")

        # オフセット(平均差)自体が時間とともに変化(ドリフト)していないかを確認
        half = n // 2
        hw_first_half_mean = hw_diff[:half].mean()
        hw_second_half_mean = hw_diff[half:].mean()
        hw_drift = hw_second_half_mean - hw_first_half_mean
        print("\n=== カメラ内部タイムスタンプのドリフト(オフセットの変化) ===")
        print(f"前半の平均差: {hw_first_half_mean:.1f}")
        print(f"後半の平均差: {hw_second_half_mean:.1f}")
        print(f"変化量: {hw_drift:.1f}  (ns想定なら /1e6 でms換算 = {hw_drift/1e6:.3f} ms)")

        # 差分がフレーム間隔(約4.4ms=227fps相当)より大きいか、
        # あるいは異常に巨大(電源投入からの独立カウント疑い)かを簡易判定する
        # ns単位と仮定した場合の目安
        one_frame_ns = 1e9 / 227.0  # 約4.4ms相当をnsで
        if hw_diff_abs.mean() > one_frame_ns * 100:
            print("\n[判定] 差分が非常に大きく、フレーム間隔の100倍を超えています。")
            print("→ 2台が共通の基準時刻を持たず、それぞれ独立にカウントしている可能性が高いです。")
            print("→ Bus同期が実際には機能していない疑いがあります。GetCamFrameSynchronizationで再確認してください。")
        else:
            print("\n[判定] 差分はフレーム間隔と近いオーダーです。同期できている可能性があります。")
            print(f"(参考: 1フレーム分の時間 ≒ {one_frame_ns:.0f} ns と仮定)")
    else:
        print("\n[参考] カメラ内部タイムスタンプは取得できませんでした(image_dataにtimestamp属性なし)。")
        print("→ 以下の結果はホスト側到着時刻ベースのため、OSのスケジューリング等のノイズを含みます。")

    print("\n=== フレーム到着時刻の差(カメラA - カメラB) ===")
    print(f"平均: {diffs_ms.mean():.3f} ms")
    print(f"標準偏差(ばらつき): {diffs_ms.std():.3f} ms")
    print(f"最小: {diffs_ms.min():.3f} ms / 最大: {diffs_ms.max():.3f} ms")

    # 前半と後半で平均がどれくらい変わるか(ドリフトの確認)
    half = n // 2
    first_half_mean = diffs_ms[:half].mean()
    second_half_mean = diffs_ms[half:].mean()
    drift = second_half_mean - first_half_mean

    print("\n=== ドリフト(時間とともにズレが広がっていないか) ===")
    print(f"前半の平均差: {first_half_mean:.3f} ms")
    print(f"後半の平均差: {second_half_mean:.3f} ms")
    print(f"変化量: {drift:.3f} ms")

    print("\n=== 判定の目安 ===")
    if abs(drift) < 1.0 and diffs_ms.std() < 2.0:
        print("→ 差がほぼ一定で、ドリフトも小さいです。同期できている可能性が高いです。")
    elif abs(drift) < 1.0:
        print("→ ドリフトは小さいですが、ばらつき(標準偏差)がやや大きいです。要確認。")
    else:
        print("→ 時間とともに差が広がっています(ドリフト)。同期できていない可能性が高いです。")


if __name__ == "__main__":
    main()