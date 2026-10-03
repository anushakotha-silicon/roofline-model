"""Command-line entry point: `roofline`."""

import argparse

from roofline.hardware import HARDWARE
from roofline.model import analyze, decode_step, prefill


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Roofline estimate for LLM inference")
    p.add_argument("--hw", choices=sorted(HARDWARE), default="h100-sxm")
    p.add_argument("--params", type=float, default=7e9, help="model parameters")
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    p.add_argument("--ctx", type=int, default=2048, help="context / prompt length")
    p.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    p.add_argument("--plot", metavar="PNG", help="save a roofline plot (needs matplotlib)")
    args = p.parse_args(argv)

    hw = HARDWARE[args.hw]
    print(f"{hw.name}: {hw.peak_tflops:.0f} TFLOP/s, {hw.mem_bw_tbs:.2f} TB/s, "
          f"ridge = {hw.ridge_point:.1f} FLOP/B\n")
    print(f"{'kernel':<28}{'AI (F/B)':>10}{'TFLOP/s':>10}{'time (ms)':>11}  bound")

    results = []
    for b in args.batch:
        if args.phase == "decode":
            k = decode_step(args.params, b, args.ctx, args.layers, args.d_model)
        else:
            k = prefill(args.params, b, args.ctx, args.layers, args.d_model)
        r = analyze(k, hw)
        results.append(r)
        print(f"{k.name:<28}{k.intensity:>10.1f}{r.attainable_flops / 1e12:>10.1f}"
              f"{r.time_s * 1e3:>11.3f}  {r.bound}")

    if args.plot:
        from roofline.plot import plot_roofline
        plot_roofline(hw, results, args.plot)
        print(f"\nSaved {args.plot}")


if __name__ == "__main__":
    main()
