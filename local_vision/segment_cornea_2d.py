#!/usr/bin/env python3
"""Segment the bright corneal arc from the dark speckle background in a 2D slice.

Intensity-based pipeline (no deep learning): denoise -> threshold -> morphology
-> keep the dominant curved component. Tuned for OCT/ultrasound-style speckle.

Usage:
    python3 segment_cornea_2d.py INPUT.png [--out-dir DIR] [--block 51] [--C 5]
"""
import argparse
import os
import cv2
import numpy as np


def segment(img_gray, block=51, C=5, min_area_frac=0.01):
    # 1. Denoise speckle while keeping the cornea edge sharp.
    den = cv2.medianBlur(img_gray, 5)
    den = cv2.bilateralFilter(den, d=7, sigmaColor=40, sigmaSpace=7)

    # 2. Threshold the bright band. Otsu for global, adaptive for uneven arcs;
    #    combine so we catch the dimmer edges of the curve.
    _, otsu = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block = block if block % 2 == 1 else block + 1
    adapt = cv2.adaptiveThreshold(
        den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -C
    )
    mask = cv2.bitwise_and(otsu, adapt)

    # 3. Morphology: close gaps along the arc, open away isolated speckle.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # 4. Keep the largest connected component (the cornea), drop background blobs.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        min_area = min_area_frac * img_gray.size
        mask = np.where((labels == keep) & (areas.max() >= min_area), 255, 0).astype(np.uint8)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--block", type=int, default=51)
    ap.add_argument("--C", type=int, default=5)
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    mask = segment(img, block=args.block, C=args.C)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input))
    stem = os.path.splitext(os.path.basename(args.input))[0]
    mask_path = os.path.join(out_dir, f"{stem}_cornea_mask.png")
    overlay_path = os.path.join(out_dir, f"{stem}_cornea_overlay.png")

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (0.4 * overlay[mask > 0] + np.array([0, 0, 153])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)
    frac = 100.0 * (mask > 0).sum() / mask.size
    print(f"cornea pixels: {frac:.1f}% of image")
    print(f"mask:    {mask_path}")
    print(f"overlay: {overlay_path}")


if __name__ == "__main__":
    main()
