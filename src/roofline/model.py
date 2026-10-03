"""Core roofline math and transformer inference workload estimates."""

from dataclasses import dataclass

from roofline.hardware import Hardware


@dataclass(frozen=True)
class Kernel:
    name: str
    flops: float
    bytes_moved: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes_moved


@dataclass(frozen=True)
class RooflineResult:
    kernel: Kernel
    hardware: Hardware
    attainable_flops: float
    time_s: float
    bound: str  # "compute" or "memory"


def analyze(kernel: Kernel, hw: Hardware) -> RooflineResult:
    attainable = min(hw.peak_flops, kernel.intensity * hw.mem_bw)
    compute_time = kernel.flops / hw.peak_flops
    memory_time = kernel.bytes_moved / hw.mem_bw
    bound = "compute" if compute_time >= memory_time else "memory"
    return RooflineResult(kernel, hw, attainable, max(compute_time, memory_time), bound)


def decode_step(
    n_params: float,
    batch: int,
    context_len: int,
    n_layers: int,
    d_model: int,
    bytes_per_param: float = 2.0,
    kv_heads_ratio: float = 1.0,
) -> Kernel:
    """One decode step for a dense transformer (weights + KV cache read once).

    kv_heads_ratio = n_kv_heads / n_heads (1.0 for MHA, <1 for GQA/MQA).
    """
    kv_bytes = 2 * n_layers * d_model * kv_heads_ratio * context_len * batch * bytes_per_param
    flops = 2 * n_params * batch + 4 * n_layers * d_model * context_len * batch
    bytes_moved = n_params * bytes_per_param + kv_bytes
    return Kernel(f"decode(b={batch}, ctx={context_len})", flops, bytes_moved)


def prefill(
    n_params: float,
    batch: int,
    seq_len: int,
    n_layers: int,
    d_model: int,
    bytes_per_param: float = 2.0,
) -> Kernel:
    """Prefill of seq_len tokens (weights read once, attention is quadratic)."""
    tokens = batch * seq_len
    flops = 2 * n_params * tokens + 2 * n_layers * d_model * seq_len * seq_len * batch
    bytes_moved = n_params * bytes_per_param + 2 * tokens * d_model * n_layers * bytes_per_param
    return Kernel(f"prefill(b={batch}, seq={seq_len})", flops, bytes_moved)
