"""Step-by-step roofline walkthrough for an H100, using only the standard library.

Run from the repo root:
    PYTHONPATH=src python3 examples/h100_roofline.py
Writes h100_roofline.svg next to where you run it.
"""

import math

from roofline import HARDWARE, analyze, decode_step, prefill

# ---------------------------------------------------------------------------
# Step 1: the hardware has two ceilings.
#   - compute ceiling: how many FLOP/s the tensor cores can do (989 TFLOP/s BF16)
#   - memory ceiling:  how many bytes/s HBM can deliver (3.35 TB/s)
# ---------------------------------------------------------------------------
hw = HARDWARE["h100-sxm"]
print(f"Step 1  {hw.name}: peak {hw.peak_tflops:.0f} TFLOP/s, HBM {hw.mem_bw_tbs} TB/s")

# ---------------------------------------------------------------------------
# Step 2: the ridge point. A kernel doing I FLOPs per byte can go no faster
# than I * bandwidth. That equals the compute peak when I = peak / bandwidth.
# Below the ridge you're memory-bound; above it you're compute-bound.
# ---------------------------------------------------------------------------
print(f"Step 2  ridge point = {hw.peak_flops:.3g} / {hw.mem_bw:.3g} "
      f"= {hw.ridge_point:.1f} FLOP/byte")

# ---------------------------------------------------------------------------
# Step 3: the roof itself. attainable(I) = min(peak, I * bandwidth).
# On a log-log plot this is a 45-degree slope that flattens at the ridge.
# ---------------------------------------------------------------------------
def roof(intensity):
    return min(hw.peak_flops, intensity * hw.mem_bw)

print(f"Step 3  roof at I=10: {roof(10) / 1e12:.1f} TFLOP/s, "
      f"at I=1000: {roof(1000) / 1e12:.1f} TFLOP/s")

# ---------------------------------------------------------------------------
# Step 4: place real workloads on it. Llama-2-7B-ish shape:
# 7e9 params, 32 layers, d_model 4096, BF16 (2 bytes/param), standard MHA.
#   decode: every step reads all the weights + the KV cache, for only
#           ~2 FLOPs per param per sequence -> very low intensity.
#   prefill: weights are read once but reused across every prompt token
#            -> intensity grows with prompt length.
# ---------------------------------------------------------------------------
P, L, D = 7e9, 32, 4096
kernels = [decode_step(P, b, 2048, L, D) for b in (1, 8, 64, 512)]
kernels += [prefill(P, 1, s, L, D) for s in (128, 512, 4096)]

print("Step 4  workloads")
results = []
for k in kernels:
    r = analyze(k, hw)
    results.append(r)
    print(f"        {k.name:<24} I={k.intensity:8.1f} F/B  "
          f"{r.attainable_flops / 1e12:6.1f} TFLOP/s  {r.time_s * 1e3:8.3f} ms  {r.bound}")

# ---------------------------------------------------------------------------
# Step 5: draw it. Map log10 values to pixel coordinates and emit SVG.
# ---------------------------------------------------------------------------
W, H = 820, 520
left, right, top, bottom = 80, 790, 50, 450
x_min, x_max = -1, 5          # 10^-1 .. 10^5 FLOP/byte
y_min, y_max = -1, 3.5        # 10^-1 .. ~3162 TFLOP/s

def px(i):
    return left + (math.log10(i) - x_min) / (x_max - x_min) * (right - left)

def py(tflops):
    return bottom - (math.log10(tflops) - y_min) / (y_max - y_min) * (bottom - top)

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
       f'<rect width="{W}" height="{H}" fill="white"/>']

# gridlines + tick labels at each decade
for e in range(x_min, x_max + 1):
    x = px(10 ** e)
    svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#eee"/>')
    svg.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" fill="#555">1e{e}</text>')
for e in range(math.ceil(y_min), math.floor(y_max) + 1):
    y = py(10 ** e)
    svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#eee"/>')
    svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{10 ** e:g}</text>')
svg.append(f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" '
           f'fill="none" stroke="#999"/>')

# the roof: sample along x and draw a polyline
pts = []
for n in range(301):
    i = 10 ** (x_min + n / 300 * (x_max - x_min))
    pts.append(f"{px(i):.1f},{py(roof(i) / 1e12):.1f}")
svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#222" stroke-width="2.5"/>')

# ridge marker
rx = px(hw.ridge_point)
svg.append(f'<line x1="{rx:.1f}" y1="{top}" x2="{rx:.1f}" y2="{bottom}" '
           f'stroke="#888" stroke-dasharray="5,4"/>')
svg.append(f'<text x="{rx + 6:.1f}" y="{bottom - 8}" fill="#666">ridge {hw.ridge_point:.0f} F/B</text>')
svg.append(f'<text x="{px(0.3):.1f}" y="{py(roof(0.3) / 1e12) - 14:.1f}" fill="#444" '
           f'transform="rotate(-36 {px(0.3):.1f} {py(roof(0.3) / 1e12) - 14:.1f})">'
           f'memory-bound: {hw.mem_bw_tbs} TB/s × I</text>')
svg.append(f'<text x="{rx + 6:.1f}" y="{py(hw.peak_tflops) - 10:.1f}" fill="#444">'
           f'compute-bound: {hw.peak_tflops:.0f} TFLOP/s</text>')

# workload points: blue = decode, orange = prefill
# hand-placed label offsets (dx, dy, anchor) so neighbouring labels don't collide
LABEL_OFFSETS = [
    (8, 16, "start"),    # decode b=1
    (8, 16, "start"),    # decode b=8
    (8, 18, "start"),    # decode b=64
    (-8, -10, "end"),    # decode b=512
    (-8, -10, "end"),    # prefill seq=128
    (8, 40, "start"),    # prefill seq=512
    (0, 22, "middle"),   # prefill seq=4096
]
for n, r in enumerate(results):
    k = r.kernel
    color = "#2a6fdb" if k.name.startswith("decode") else "#e8833a"
    x, y = px(k.intensity), py(r.attainable_flops / 1e12)
    svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="white"/>')
    dx, dy, anchor = LABEL_OFFSETS[n]
    svg.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" '
               f'fill="{color}" font-size="11">{k.name}</text>')

# titles and legend
svg.append(f'<text x="{left}" y="28" font-size="16" font-weight="bold" fill="#222">'
           f'Roofline: {hw.name}, 7B model, BF16</text>')
svg.append(f'<text x="{(left + right) / 2}" y="{H - 30}" text-anchor="middle" fill="#333">'
           f'Arithmetic intensity (FLOP / byte)</text>')
svg.append(f'<text x="22" y="{(top + bottom) / 2}" text-anchor="middle" fill="#333" '
           f'transform="rotate(-90 22 {(top + bottom) / 2})">Attainable TFLOP/s</text>')
svg.append(f'<circle cx="{right - 150}" cy="{H - 34}" r="5" fill="#2a6fdb"/>'
           f'<text x="{right - 140}" y="{H - 30}" fill="#333">decode</text>'
           f'<circle cx="{right - 80}" cy="{H - 34}" r="5" fill="#e8833a"/>'
           f'<text x="{right - 70}" y="{H - 30}" fill="#333">prefill</text>')
svg.append("</svg>")

with open("h100_roofline.svg", "w") as f:
    f.write("\n".join(svg))
print("Step 5  wrote h100_roofline.svg")
