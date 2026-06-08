#!/usr/bin/env python3
"""Synthesize a cornea-like test slice: a bright curved arc over speckle."""
import cv2
import numpy as np

H = W = 240
rng = np.random.default_rng(7)

# Speckle background: brown-ish texture -> grayscale base around mid-tone.
base = rng.normal(120, 28, (H, W)).clip(0, 255)

# Bright curved arc (the "cornea"): an annulus segment.
yy, xx = np.mgrid[0:H, 0:W]
cx, cy = 120, 320          # circle center below the frame -> upward-curving arc
r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
arc = (np.abs(r - 250) < 14) & (xx > 40) & (xx < 200)
band = np.zeros((H, W))
band[arc] = 1.0
band = cv2.GaussianBlur(band, (0, 0), 3)          # soft edges
img = base + band * 90 * rng.normal(1.0, 0.15, (H, W))  # speckle on the band too
img = cv2.GaussianBlur(img, (3, 3), 0).clip(0, 255).astype(np.uint8)

cv2.imwrite("/home/zhuojian/Desktop/Integration/local_vision/test_cornea.png", img)
print("wrote test_cornea.png")
