import os
import psutil
import pytest
import GPUtil


@pytest.fixture
def test_memory_usage():
    """メモリ使用量をチェックする"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    pytest.main(["day5/演習3/tests/test_model.py"])
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_used = final_memory - initial_memory
    assert memory_used < 1024, f"メモリ使用量が高すぎます: {memory_used:.2f}MB"


@pytest.fixture
def test_cpu_usage():
    """CPU使用率をチェックする"""
    process = psutil.Process(os.getpid())
    initial_cpu = process.cpu_percent(interval=None)
    pytest.main(["day5/演習3/tests/test_model.py"])
    final_cpu = process.cpu_percent(interval=None)
    cpu_used = final_cpu - initial_cpu
    assert cpu_used < 80, f"CPU使用率が高すぎます: {cpu_used:.2f}%"


@pytest.fixture
def test_disk_usage():
    """ディスク使用量をチェックする"""
    initial_disk = psutil.disk_usage("/").used / 1024 / 1024
    pytest.main(["day5/演習3/tests/test_model.py"])
    final_disk = psutil.disk_usage("/").used / 1024 / 1024
    disk_used = final_disk - initial_disk
    assert disk_used < 2048, f"ディスク使用量が高すぎます: {disk_used:.2f}MB"


@pytest.fixture
def test_network_usage():
    """ネットワーク使用量をチェックする"""
    initial_net = (
        psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    )
    pytest.main(["day5/演習3/tests/test_model.py"])
    final_net = (
        psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    )
    net_used = final_net - initial_net
    assert net_used < 1024 * 1024, f"ネットワーク使用量が高すぎます: {net_used:.2f}MB"


@pytest.fixture
def test_storage_usage():
    """ストレージ使用量をチェックする"""
    initial_storage = psutil.disk_usage("/").used / 1024 / 1024
    pytest.main(["day5/演習3/tests/test_model.py"])
    final_storage = psutil.disk_usage("/").used / 1024 / 1024
    storage_used = final_storage - initial_storage
    assert storage_used < 2048, f"ストレージ使用量が高すぎます: {storage_used:.2f}MB"


@pytest.fixture
def test_gpu_usage():
    """GPU使用率をチェックする"""
    try:
        gpus = GPUtil.getGPUs()
        initial_gpu = sum(gpu.memoryUsed for gpu in gpus)
        pytest.main(["day5/演習3/tests/test_model.py"])
        final_gpu = sum(gpu.memoryUsed for gpu in gpus)
        gpu_used = final_gpu - initial_gpu
        assert gpu_used < 1024, f"GPU使用量が高すぎます: {gpu_used:.2f}MB"
    except ImportError:
        pytest.skip(
            "GPUtilがインストールされていないため、GPU使用量のチェックをスキップします"
        )
