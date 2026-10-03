"""Accelerator specs used by the roofline model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_tflops: float  # dense FP16/BF16 TFLOP/s
    mem_bw_tbs: float  # HBM bandwidth, TB/s

    @property
    def peak_flops(self) -> float:
        return self.peak_tflops * 1e12

    @property
    def mem_bw(self) -> float:
        return self.mem_bw_tbs * 1e12

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) where compute and memory bounds meet."""
        return self.peak_flops / self.mem_bw


# Published dense BF16 peaks (no sparsity). Edit or extend as needed.
HARDWARE = {
    "a100-80g": Hardware("NVIDIA A100 80GB SXM", 312.0, 2.039),
    "h100-sxm": Hardware("NVIDIA H100 SXM", 989.0, 3.35),
    "h200": Hardware("NVIDIA H200", 989.0, 4.8),
    "mi300x": Hardware("AMD MI300X", 1307.0, 5.3),
}
