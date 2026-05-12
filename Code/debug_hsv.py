"""Debug: analyze HSV distribution of the conveyor image."""
import cv2, numpy as np
img = cv2.imread('d:/FYP/relevant_frames_test/peak_000074_3340ms_area47101.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Center crop (likely mango region)
cy, cx = img.shape[0]//2, img.shape[1]//2
r = 80
mango_h = h[cy-r:cy+r, cx-r:cx+r]
mango_s = s[cy-r:cy+r, cx-r:cx+r]
mango_v = v[cy-r:cy+r, cx-r:cx+r]

print("=== Center mango region (approx) ===")
print(f"H: min={mango_h.min()}, max={mango_h.max()}, mean={mango_h.mean():.1f}, median={np.median(mango_h):.1f}")
print(f"S: min={mango_s.min()}, max={mango_s.max()}, mean={mango_s.mean():.1f}")
print(f"V: min={mango_v.min()}, max={mango_v.max()}, mean={mango_v.mean():.1f}")

# Full image stats
print("\n=== Full image ===")
print(f"H: min={h.min()}, max={h.max()}, mean={h.mean():.1f}")
print(f"S: min={s.min()}, max={s.max()}, mean={s.mean():.1f}")
print(f"V: min={v.min()}, max={v.max()}, mean={v.mean():.1f}")

# Bright region (mango body likely)
bright = v > 100
print(f"\n=== Bright pixels (V>100): {bright.sum()} px ===")
if bright.sum() > 0:
    print(f"H: min={h[bright].min()}, max={h[bright].max()}, mean={h[bright].mean():.1f}")
    print(f"S: min={s[bright].min()}, max={s[bright].max()}, mean={s[bright].mean():.1f}")

# Very bright (V>140)
vbright = v > 140
print(f"\n=== Very bright (V>140): {vbright.sum()} px ===")
if vbright.sum() > 0:
    print(f"H: min={h[vbright].min()}, max={h[vbright].max()}, mean={h[vbright].mean():.1f}")
    print(f"S: min={s[vbright].min()}, max={s[vbright].max()}, mean={s[vbright].mean():.1f}")

# How many pixels pass current hue filter 15-95
hue_pass = (h >= 15) & (h <= 95) & (s >= 30) & (v >= 80)
print(f"\n=== Pass hue filter (H:15-95, S>30, V>80): {hue_pass.sum()} px ===")

# Wider hue 10-110
hue_pass2 = (h >= 10) & (h <= 110) & (s >= 20) & (v >= 60)
print(f"=== Wider filter (H:10-110, S>20, V>60): {hue_pass2.sum()} px ===")
