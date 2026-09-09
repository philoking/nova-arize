#!/usr/bin/env python3
"""Generate Nova's PWA icons with the standard library only (zlib + struct).

Draws a rounded-square gradient tile with the Nova mark — a four-point
starburst ("nova") in helmet-gold with a glowing blue-white core, matching the
SVG logo in the header and the favicon (frontend/icons/nova-mark.svg).
Run from the repo root:  python3 scripts/make_icons.py
"""

from __future__ import annotations

import math
import os
import struct
import zlib

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "icons")

BG_TOP = (11, 16, 32)        # --bg  (dark tile so the gold star pops)
BG_BOTTOM = (18, 26, 51)     # --bg-2
GOLD_TOP = (255, 221, 130)   # star, lit edge
GOLD_BOTTOM = (240, 160, 30) # star, shadow edge
CORE_INNER = (255, 255, 255) # Nova-force core
CORE_OUTER = (135, 175, 255) # blue energy halo


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _rounded(x: int, y: int, w: int, h: int, r: int) -> bool:
    """Inside a rounded rectangle occupying the whole canvas?"""
    if x < r and y < r:
        return (r - x) ** 2 + (r - y) ** 2 <= r * r
    if x >= w - r and y < r:
        return (x - (w - r)) ** 2 + (r - y) ** 2 <= r * r
    if x < r and y >= h - r:
        return (r - x) ** 2 + (y - (h - r)) ** 2 <= r * r
    if x >= w - r and y >= h - r:
        return (x - (w - r)) ** 2 + (y - (h - r)) ** 2 <= r * r
    return True


def _star_bound(dx: float, dy: float, outer: float, inner: float, phase: float = 0.0) -> float:
    """Boundary radius of a four-point star at the angle of (dx, dy).

    Tips sit at `outer`, the notches between them at `inner`; the edge is linear
    between the two. `phase` (degrees) rotates the star — used for the diagonal
    ray set that turns the 4-point star into an 8-point nova burst.
    """
    ang = (math.degrees(math.atan2(dy, dx)) - phase) % 90.0
    if ang <= 45.0:
        return outer + (inner - outer) * (ang / 45.0)
    return inner + (outer - inner) * ((ang - 45.0) / 45.0)


def make_icon(size: int, maskable: bool = False) -> bytes:
    # Maskable icons need a safe zone: shrink the glyph and use no corner radius.
    radius = 0 if maskable else size // 5
    cx, cy = size / 2, size / 2
    scale = 0.80 if maskable else 1.0

    outer = size * 0.47 * scale           # primary star tip radius
    inner = size * 0.115 * scale          # notch radius — deep, for sharp points
    ray_outer = size * 0.25 * scale       # diagonal rays: short + thin spikes
    ray_inner = size * 0.045 * scale
    core = size * 0.135 * scale           # glowing core radius
    top_star = cy - outer                 # for the vertical gold gradient

    raw = bytearray()
    for y in range(size):
        raw.append(0)  # PNG filter byte: none
        for x in range(size):
            inside = _rounded(x, y, size, size, radius)
            if not inside:
                raw.extend((0, 0, 0, 0))  # transparent outside the tile
                continue
            r, g, b = _lerp(BG_TOP, BG_BOTTOM, y / size)

            dx, dy = x - cx, y - cy
            dist = (dx * dx + dy * dy) ** 0.5

            in_star = (
                dist <= _star_bound(dx, dy, outer, inner)
                or dist <= _star_bound(dx, dy, ray_outer, ray_inner, phase=45.0)
            )
            if in_star:
                t = min(max((y - top_star) / (2 * outer), 0.0), 1.0)
                r, g, b = _lerp(GOLD_TOP, GOLD_BOTTOM, t)

            # Glowing blue-white core on top of the star.
            if dist <= core:
                r, g, b = _lerp(CORE_INNER, CORE_OUTER, min(dist / core, 1.0))

            raw.extend((r, g, b, 255))

    return _png(size, size, bytes(raw))


def _png(width: int, height: int, raw: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    idat = zlib.compress(raw, 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = [
        ("icon-192.png", 192, False),
        ("icon-512.png", 512, False),
        ("icon-maskable-512.png", 512, True),
    ]
    for name, size, maskable in targets:
        path = os.path.join(OUT_DIR, name)
        with open(path, "wb") as f:
            f.write(make_icon(size, maskable))
        print(f"wrote {path} ({size}x{size}{' maskable' if maskable else ''})")


if __name__ == "__main__":
    main()
