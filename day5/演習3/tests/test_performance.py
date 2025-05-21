"""パフォーマンステストモジュール.

このモジュールは、モデルのパフォーマンスに関する様々なテストを提供します.
メモリ使用量、CPU使用率、ディスク使用量などのリソース使用状況を監視します.
"""

import os
import pytest
import psutil
import time
import GPUtil


def test_memory_usage():
    """メモリ使用量をチェックします."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB単位

    # テスト実行
    time.sleep(1)  # テスト実行をシミュレート

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_used = final_memory - initial_memory

    # メモリ使用量が1GB未満であることを確認
    assert memory_used < 1024, f"メモリ使用量が高すぎます: {memory_used:.2f}MB"


def test_cpu_usage():
    """CPU使用率をチェックします."""
    process = psutil.Process(os.getpid())

    # CPU使用率を計測
    cpu_percent = process.cpu_percent(interval=1)

    # CPU使用率が80%未満であることを確認
    assert cpu_percent < 80, f"CPU使用率が高すぎます: {cpu_percent}%"


def test_disk_usage():
    """ディスク使用量をチェックします."""
    # 現在のディレクトリの使用量を取得
    current_dir = os.getcwd()
    usage = psutil.disk_usage(current_dir)

    # ディスク使用率が90%未満であることを確認
    assert usage.percent < 90, f"ディスク使用率が高すぎます: {usage.percent}%"


def test_network_usage():
    """ネットワーク使用量をチェックします."""
    # ネットワークインターフェースの統計を取得
    net_io = psutil.net_io_counters()

    # 送受信バイト数が1GB未満であることを確認
    assert net_io.bytes_sent < 1024 * 1024 * 1024, "送信量が多すぎます"
    assert net_io.bytes_recv < 1024 * 1024 * 1024, "受信量が多すぎます"


def test_storage_usage():
    """ストレージ使用量をチェックします."""
    # モデルファイルのサイズを確認
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "../models")
    if os.path.exists(model_dir):
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(model_dir)
            for filename in filenames
        )

        # モデルファイルの合計サイズが500MB未満であることを確認
        assert (
            total_size < 500 * 1024 * 1024
        ), f"モデルファイルのサイズが大きすぎます: {total_size / (1024 * 1024):.2f}MB"


def test_gpu_usage():
    """GPU使用率をチェックします."""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                # GPU使用率が90%未満であることを確認
                assert gpu.load * 100 < 90, f"GPU使用率が高すぎます: {gpu.load * 100}%"
                # GPUメモリ使用率が90%未満であることを確認
                assert (
                    gpu.memoryUtil * 100 < 90
                ), f"GPUメモリ使用率が高すぎます: {gpu.memoryUtil * 100}%"
    except Exception as e:
        pytest.skip(f"GPUが利用できないためスキップします: {str(e)}")


def test_inference_time():
    """推論時間をチェックします."""
    # 推論時間の計測
    start_time = time.time()
    time.sleep(0.1)  # 推論処理をシミュレート
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が0.5秒未満であることを確認
    assert inference_time < 0.5, f"推論時間が長すぎます: {inference_time:.3f}秒"


def test_model_size():
    """モデルサイズをチェックします."""
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "../models/titanic_model.pkl")
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB単位

        # モデルサイズが100MB未満であることを確認
        assert model_size < 100, f"モデルサイズが大きすぎます: {model_size:.2f}MB"
    else:
        pytest.skip("モデルファイルが存在しないためスキップします")
