import pytest

from roofline import HARDWARE, Kernel, analyze, decode_step, prefill


def test_ridge_point_h100():
    assert HARDWARE["h100-sxm"].ridge_point == pytest.approx(989e12 / 3.35e12)


def test_memory_bound_kernel():
    hw = HARDWARE["h100-sxm"]
    r = analyze(Kernel("k", flops=1e9, bytes_moved=1e9), hw)
    assert r.bound == "memory"
    assert r.attainable_flops == pytest.approx(hw.mem_bw)


def test_compute_bound_kernel():
    hw = HARDWARE["h100-sxm"]
    r = analyze(Kernel("k", flops=1e15, bytes_moved=1e9), hw)
    assert r.bound == "compute"
    assert r.attainable_flops == hw.peak_flops


def test_decode_batch1_is_memory_bound():
    k = decode_step(7e9, batch=1, context_len=2048, n_layers=32, d_model=4096)
    assert analyze(k, HARDWARE["h100-sxm"]).bound == "memory"


def test_large_prefill_is_compute_bound():
    k = prefill(7e9, batch=8, seq_len=4096, n_layers=32, d_model=4096)
    assert analyze(k, HARDWARE["h100-sxm"]).bound == "compute"
