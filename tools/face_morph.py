"""GA3 Q7 — morph one face into another.

Piecewise-affine (Delaunay) morph between hand-placed corresponding landmarks,
which keeps the eyes, nose and mouth travelling together instead of the ghosting
a plain cross-dissolve produces. Frames are rendered to PNG and encoded by
ffmpeg under the 100,000-byte limit.

Usage: python tools/face_morph.py <from.jpg> <to.jpg> <out.mp4>
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

WIDTH, HEIGHT = 360, 480
FPS = 24
MORPH_SECONDS = 2.4
HOLD_SECONDS = 0.35

# Corresponding landmarks, normalised to each photograph, in matching order.
FROM_LANDMARKS = [
    (0.145, 0.300), (0.195, 0.300), (0.245, 0.295),        # image-left eye
    (0.330, 0.265), (0.370, 0.260), (0.410, 0.258),        # image-right eye
    (0.270, 0.300), (0.280, 0.375), (0.245, 0.390), (0.330, 0.360),   # nose
    (0.220, 0.450), (0.290, 0.460), (0.360, 0.440),        # mouth corners/centre
    (0.290, 0.432), (0.300, 0.492),                        # upper and lower lip
    (0.330, 0.580),                                        # chin
    (0.120, 0.430), (0.450, 0.420),                        # jaw
    (0.100, 0.320), (0.480, 0.270),                        # cheeks
    (0.260, 0.200), (0.280, 0.060), (0.530, 0.330),        # forehead, hair, ear
]
TO_LANDMARKS = [
    (0.265, 0.340), (0.315, 0.340), (0.365, 0.340),
    (0.465, 0.340), (0.515, 0.340), (0.565, 0.340),
    (0.400, 0.360), (0.400, 0.435), (0.360, 0.440), (0.450, 0.430),
    (0.330, 0.510), (0.405, 0.515), (0.480, 0.510),
    (0.405, 0.497), (0.405, 0.537),
    (0.410, 0.620),
    (0.260, 0.500), (0.570, 0.490),
    (0.240, 0.360), (0.600, 0.350),
    (0.400, 0.270), (0.400, 0.100), (0.620, 0.420),
]
# Frame border, so the background stays put while the face moves.
BORDER = [(x, y) for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)]


def load(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def points(landmarks) -> np.ndarray:
    pts = [(x * (WIDTH - 1), y * (HEIGHT - 1)) for x, y in list(landmarks) + BORDER]
    return np.array(pts, dtype=np.float32)


def sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``image`` at float coordinates."""
    height, width, _ = image.shape
    xs = np.clip(xs, 0, width - 1.001)
    ys = np.clip(ys, 0, height - 1.001)
    x0, y0 = np.floor(xs).astype(np.int32), np.floor(ys).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = (xs - x0)[:, None], (ys - y0)[:, None]
    return (
        image[y0, x0] * (1 - fx) * (1 - fy)
        + image[y0, x1] * fx * (1 - fy)
        + image[y1, x0] * (1 - fx) * fy
        + image[y1, x1] * fx * fy
    )


def warp(image: np.ndarray, source: np.ndarray, target: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    out = np.zeros_like(image)
    filled = np.zeros((HEIGHT, WIDTH), dtype=bool)
    grid_y, grid_x = np.mgrid[0:HEIGHT, 0:WIDTH]
    for tri in triangles:
        dst = target[tri]
        src = source[tri]
        min_x, min_y = np.floor(dst.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(dst.max(axis=0)).astype(int)
        min_x, min_y = max(min_x, 0), max(min_y, 0)
        max_x, max_y = min(max_x + 1, WIDTH), min(max_y + 1, HEIGHT)
        if min_x >= max_x or min_y >= max_y:
            continue
        xs = grid_x[min_y:max_y, min_x:max_x].ravel()
        ys = grid_y[min_y:max_y, min_x:max_x].ravel()
        matrix = np.array(
            [[dst[1, 0] - dst[0, 0], dst[2, 0] - dst[0, 0]],
             [dst[1, 1] - dst[0, 1], dst[2, 1] - dst[0, 1]]],
            dtype=np.float64,
        )
        determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        if abs(determinant) < 1e-8:
            continue
        inverse = np.array([[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]]) / determinant
        offset_x = xs - dst[0, 0]
        offset_y = ys - dst[0, 1]
        u = inverse[0, 0] * offset_x + inverse[0, 1] * offset_y
        v = inverse[1, 0] * offset_x + inverse[1, 1] * offset_y
        inside = (u >= -0.001) & (v >= -0.001) & (u + v <= 1.001)
        if not inside.any():
            continue
        u, v = u[inside], v[inside]
        px, py = xs[inside], ys[inside]
        src_x = src[0, 0] + u * (src[1, 0] - src[0, 0]) + v * (src[2, 0] - src[0, 0])
        src_y = src[0, 1] + u * (src[1, 1] - src[0, 1]) + v * (src[2, 1] - src[0, 1])
        out[py, px] = sample(image, src_x, src_y)
        filled[py, px] = True
    if not filled.all():
        out[~filled] = image[~filled]
    return out


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def main() -> int:
    src_path, dst_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    image_a, image_b = load(src_path), load(dst_path)
    points_a, points_b = points(FROM_LANDMARKS), points(TO_LANDMARKS)
    triangles = Delaunay((points_a + points_b) / 2).simplices

    morph_frames = int(FPS * MORPH_SECONDS)
    hold_frames = int(FPS * HOLD_SECONDS)
    with tempfile.TemporaryDirectory() as workdir:
        index = 0

        def write(frame: np.ndarray) -> None:
            nonlocal index
            Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(
                os.path.join(workdir, f"f{index:04d}.png")
            )
            index += 1

        for _ in range(hold_frames):
            write(image_a)
        for step in range(1, morph_frames):
            t = ease(step / morph_frames)
            target = (1 - t) * points_a + t * points_b
            warped_a = warp(image_a, points_a, target, triangles)
            warped_b = warp(image_b, points_b, target, triangles)
            write((1 - t) * warped_a + t * warped_b)
        for _ in range(hold_frames):
            write(image_b)

        for crf in (24, 26, 28, 30, 32, 34, 36, 38, 40, 42):
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                 "-i", os.path.join(workdir, "f%04d.png"),
                 "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-vf", "format=yuv420p", out_path],
                check=True,
            )
            size = os.path.getsize(out_path)
            print(f"crf {crf}: {size} bytes")
            if size < 100_000:
                print(f"wrote {out_path} ({size} bytes, {index} frames)")
                return 0
    print("could not reach the size limit", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
