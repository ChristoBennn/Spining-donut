#!/usr/bin/env python3
"""A spinning doughnut drawn in text, after Andy Sloane's 2006 "donut.c".

The shape is a genuine 3D torus, not a fake: every frame we

  1. take a fixed set of surface points on the torus (and the outward normal at
     each point),
  2. rotate them around the X axis (angle ``a``) and then the Z axis (angle
     ``b``),
  3. perspective-project them onto a grid of character cells by dividing by the
     depth (that division is what makes the far side look smaller),
  4. light each point by dotting its rotated normal with a fixed light
     direction, and pick a character from a dim-to-bright ramp, and
  5. keep only the nearest surface per cell with a z-buffer, so the donut looks
     solid instead of transparent.

Steps 2-5 are vectorised with numpy when it is installed; without it the same
maths runs in a plain Python loop, and the auto-tuner below thins the sampling to
keep up.  Either way the number of samples starts from the terminal size and is
then adjusted every frame to hold --fps, so an 80-column window and a 400-column
one both stay smooth.

Run it:

    python3 donut.py              # classic ASCII, sized to your terminal
    python3 donut.py --color      # 24-bit colour
    python3 donut.py --fps 60     # aim higher; the tuner thins the sampling to suit
    python3 donut.py -h           # all the knobs
"""

from __future__ import annotations

import argparse
import functools
import math
import shutil
import sys
import time
from math import ceil

try:
    import numpy as np
except ImportError:  # still runs, just slower
    np = None

# The vectorised path works in single precision.  For a grid of character cells
# the error that introduces is about 1e-5 of a cell, and it means the cell
# indices can be int32: casting float32 -> int32 costs roughly a twentieth of
# what float64 -> int64 does, which is the single biggest win in the frame loop.
DTYPE = np.float32 if np is not None else None

TAU = math.tau

# Geometry: tube radius and the distance from the donut's centre to the tube centre.
R1 = 1.0
R2 = 2.0
# Distance from the viewer to the donut's centre.  R1 + R2 must stay below this.
K2 = 5.0

# Brightness ramp, dimmest to brightest.
CHARSET = ".,-~:;=!*#$@"

# Samples per character cell the auto-tuner will not thin below.  Below about
# three the surface starts to show speckle where no sample reached a cell.
MIN_QUALITY = 3.0

# Direction the light comes from, normalised.
_LIGHT = (0.0, 1.0, -1.0)
_LIGHT_LEN = math.sqrt(sum(c * c for c in _LIGHT))
LX, LY, LZ = (c / _LIGHT_LEN for c in _LIGHT)

# Colour stops for --color, from the dark side of the donut to the lit side.
GRADIENT = (
    (0.00, (24, 0, 72)),
    (0.35, (176, 28, 94)),
    (0.62, (255, 112, 24)),
    (0.85, (255, 214, 80)),
    (1.00, (255, 255, 236)),
)

# " " for a blank cell, then one character per brightness step.
CHAR_LUT = np.array(list(" " + CHARSET)) if np is not None else None

ENGINE = "numpy" if np is not None else "python"


def ramp_color(t):
    """Interpolate the colour gradient at 0.0 <= t <= 1.0."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    for (p0, c0), (p1, c1) in zip(GRADIENT, GRADIENT[1:]):
        if t <= p1:
            f = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return tuple(int(a + (b - a) * f) for a, b in zip(c0, c1))
    return GRADIENT[-1][1]


@functools.cache
def color_table():
    """The 256 terminal colour escapes for --color, built once, indexed by brightness."""
    return ["\x1b[38;2;%d;%d;%dm" % ramp_color(i / 255.0) for i in range(256)]


def auto_steps(cells, quality):
    """Pick sampling steps sized to the number of character cells to fill.

    ``quality`` is roughly how many surface samples to aim at per cell; theta is
    sampled more finely than phi because the outer ring of the torus covers more
    of the screen than the inner one. Without this, a fixed step is either too
    coarse on a big terminal or wasteful on a small one.
    """
    ratio = 0.3  # d_phi / d_theta
    n_theta = math.sqrt(max(cells * quality, 1.0) * ratio)
    return TAU / n_theta, TAU / (n_theta / ratio)


# ---------------------------------------------------------------------------
# Sampling the torus
# ---------------------------------------------------------------------------
# theta runs around the tube's cross-section, phi around the hole, and the
# samples are laid out as a grid: one row per theta, one column per phi.  Each
# point is (x, y, z) followed by the outward surface normal (nx, ny, nz), so the
# per-frame loop only has to rotate.  The grid layout matters because it lets the
# tuner below thin the sampling by taking a stride, which is the same as having
# sampled more coarsely in the first place, but free.


def torus_grid_python(r1, r2, d_theta, d_phi):
    rows = []
    for ti in range(ceil(TAU / d_theta)):
        theta = ti * d_theta
        ct, st = math.cos(theta), math.sin(theta)
        # How far this slice of the tube sits from the donut's axis.
        ring = r2 + r1 * ct
        row = []
        for pi in range(ceil(TAU / d_phi)):
            phi = pi * d_phi
            cp, sp = math.cos(phi), math.sin(phi)
            row.append((
                ring * cp, ring * sp, r1 * st,  # position on the surface
                ct * cp, ct * sp, st,           # outward normal there
            ))
        rows.append(row)
    return rows


def torus_grid_numpy(r1, r2, d_theta, d_phi):
    theta = (np.arange(ceil(TAU / d_theta), dtype=DTYPE) * DTYPE(d_theta))[:, None]
    phi = (np.arange(ceil(TAU / d_phi), dtype=DTYPE) * DTYPE(d_phi))[None, :]
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    ring = r2 + r1 * ct          # (n_theta, 1), broadcast against phi
    nx = ct * cp                 # each of these is (n_theta, n_phi)
    ny = ct * sp
    return np.stack([
        ring * cp,
        ring * sp,
        np.broadcast_to(r1 * st, nx.shape),
        nx,
        ny,
        np.broadcast_to(st, nx.shape),
    ], axis=-1)


def thin_python(grid, rows, cols):
    """Keep every ``rows``-th theta and ``cols``-th phi sample."""
    return [point for row in grid[::rows] for point in row[::cols]]


def thin_numpy(grid, rows, cols):
    return grid[::rows, ::cols].reshape(-1, 6)


def grid_shape(grid):
    """(n_theta, n_phi) for a sample grid from either engine."""
    shape = getattr(grid, "shape", None)  # numpy arrays have one, lists of rows don't
    return shape[:2] if shape is not None else (len(grid), len(grid[0]))


def torus_grid_flat(grid):
    """The grid as a plain sequence of points."""
    return thin(grid, 1, 1)


class QualityTuner:
    """Decides how much of the sampled torus each frame gets to use.

    ``--quality`` sets how densely the torus is sampled, and that is this
    tuner's ceiling: it only ever thins the sampling down, to keep a frame inside
    the time budget that ``--fps`` implies.  Thin sampling is exactly what a
    coarser ``--dtheta``/``--dphi`` would have produced, so the ladder of stride
    pairs is sorted by sample count and the tuner walks along it in response to
    how long the last frame actually took.  It stops at ``MIN_QUALITY`` samples
    per cell, where the surface would start to show gaps.
    """

    # Coarsest stride considered on each axis.  Strides multiply, so the ladder
    # steps down through densities of 1, 1/2, 1/3 ... of the sampled grid.
    MAX_STRIDE = 12

    def __init__(self, grid, cells):
        n_theta, n_phi = grid_shape(grid)
        full = n_theta * n_phi
        floor = min(max(MIN_QUALITY * cells, 1.0), full)

        # One ladder entry per attainable sample count, densest first.
        options = {}
        for rows in range(1, self.MAX_STRIDE + 1):
            for cols in range(1, self.MAX_STRIDE + 1):
                count = -(-n_theta // rows) * -(-n_phi // cols)
                if floor <= count <= full:
                    options.setdefault(int(count), (rows, cols))
        self.ladder = sorted(options.items(), reverse=True)
        self.grid = grid
        self.index = 0
        self.points = self._thinned(0)

    def _thinned(self, index):
        rows, cols = self.ladder[index][1]
        return thin(self.grid, rows, cols)

    @property
    def count(self):
        """Samples per frame at the current setting."""
        return self.ladder[self.index][0]

    def observe(self, seconds, budget):
        """Feed in one frame's duration; True if the density moved.

        A frame only counts as out of budget outside a tolerance band, so normal
        jitter cannot make the quality flap around.
        """
        over, under = seconds > budget * 1.05, seconds < budget * 0.55
        if not (over or under):
            return False

        # Cost is close to proportional to the sample count, so scale the count
        # we used by how far off the budget the frame was.  Aiming slightly under
        # leaves room for the sleep that caps the frame rate.
        wanted = self.count * 0.9 * budget / max(seconds, 1e-9)
        new = self._for_count(wanted)
        new = max(new, self.index + 1) if over else min(new, self.index - 1)
        new = min(max(new, 0), len(self.ladder) - 1)
        if new == self.index:
            return False
        self.index = new
        self.points = self._thinned(new)
        return True

    def _for_count(self, wanted):
        """Ladder index of the densest setting no denser than ``wanted``."""
        for index, (count, _) in enumerate(self.ladder):
            if count <= wanted:
                return index
        return len(self.ladder) - 1


torus_grid = torus_grid_numpy if np is not None else torus_grid_python
thin = thin_numpy if np is not None else thin_python


# ---------------------------------------------------------------------------
# Rendering one frame: the same steps, once per engine
# ---------------------------------------------------------------------------


def _rows_from_grids(char_idx, levels, width, height, color):
    """Turn the flat per-cell character levels into the text of one frame."""
    grid = char_idx.reshape(height, width)
    if not color:
        return "\n".join("".join(row) for row in CHAR_LUT[grid + 1].tolist())

    table = color_table()
    rows = []
    for char_row, level_row in zip(grid.tolist(), levels.reshape(height, width).tolist()):
        rows.append("".join(" " if c < 0 else table[l] + CHARSET[c]
                            for c, l in zip(char_row, level_row)))
    return "\n".join(rows) + "\x1b[0m"


def render_frame_python(points, a, b, width, height, scale, color):
    """Render one frame at rotation angles ``a`` and ``b`` and return it as text."""
    cells = [" "] * (width * height)
    zbuf = [0.0] * (width * height)

    sa, ca = math.sin(a), math.cos(a)
    sb, cb = math.sin(b), math.cos(b)
    half_w, half_h = width / 2.0, height / 2.0
    # Terminal cells are roughly twice as tall as they are wide, so squash y.
    y_scale = scale / 2.0
    table = color_table() if color else None

    for x, y, z, nx, ny, nz in points:
        # Rotate around X by a, then around Z by b.
        y1 = y * ca - z * sa
        z1 = y * sa + z * ca
        x2 = x * cb - y1 * sb
        y2 = x * sb + y1 * cb

        depth = z1 + K2
        inv = 1.0 / depth  # perspective: closer points project further out

        px = int(half_w + scale * x2 * inv)
        py = int(half_h - y_scale * y2 * inv)
        if not (0 <= px < width and 0 <= py < height):
            continue

        i = py * width + px
        if inv < zbuf[i]:  # something nearer is already here
            continue
        # (An exact tie means two samples sit at the same depth, which happens
        # on mirror-image points; the later sample wins, exactly as it does in
        # the numpy renderer, so both engines agree character for character.)

        # Rotate the normal the same way, then light it.
        ny1 = ny * ca - nz * sa
        nz1 = ny * sa + nz * ca
        nx2 = nx * cb - ny1 * sb
        ny2 = nx * sb + ny1 * cb
        lum = nx2 * LX + ny2 * LY + nz1 * LZ
        lum = 0.0 if lum < 0.0 else (1.0 if lum > 1.0 else lum)

        zbuf[i] = inv
        level = int(lum * (len(CHARSET) - 1))
        cells[i] = table[int(lum * 255)] + CHARSET[level] if color else CHARSET[level]

    frame = "\n".join("".join(cells[y * width:(y + 1) * width]) for y in range(height))
    return frame + "\x1b[0m" if color else frame


def render_frame_numpy(points, a, b, width, height, scale, color):
    sa, ca = math.sin(a), math.cos(a)
    sb, cb = math.sin(b), math.cos(b)
    half_w, half_h = width / 2.0, height / 2.0
    y_scale = scale / 2.0

    # Rotate around X by a, then around Z by b, then project.  All of this is
    # the same arithmetic as render_frame_python, just applied to whole arrays.
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    y1 = y * ca - z * sa
    z1 = y * sa + z * ca
    inv = 1.0 / (z1 + K2)
    x2 = x * cb - y1 * sb
    y2 = x * sb + y1 * cb

    px = (half_w + scale * x2 * inv).astype(np.int32)
    py = (half_h - y_scale * y2 * inv).astype(np.int32)
    inside = np.flatnonzero((px >= 0) & (px < width) & (py >= 0) & (py < height))

    # Flatten to cell indices and keep the nearest surface in each cell.  The
    # scatter-max is the whole z-buffer: one pass instead of a Python loop.
    idx = py[inside] * width + px[inside]
    depth = inv[inside]
    nearest = np.zeros(width * height, dtype=DTYPE)
    np.maximum.at(nearest, idx, depth)
    front = depth >= nearest[idx]  # ties cost nothing: equal depth, equal result

    idx = idx[front]
    visible = inside[front]

    # Light only the points that survived, using the same rotations again.
    normals = points[visible, 3:]
    ny1 = normals[:, 1] * ca - normals[:, 2] * sa
    nz1 = normals[:, 1] * sa + normals[:, 2] * ca
    nx2 = normals[:, 0] * cb - ny1 * sb
    ny2 = normals[:, 0] * sb + ny1 * cb
    lum = np.clip(nx2 * LX + ny2 * LY + nz1 * LZ, 0.0, 1.0)

    char_idx = np.full(width * height, -1, dtype=np.int8)
    char_idx[idx] = (lum * (len(CHARSET) - 1)).astype(np.int8)

    levels = None
    if color:
        levels = np.zeros(width * height, dtype=np.uint8)
        levels[idx] = (lum * 255.0).astype(np.uint8)

    return _rows_from_grids(char_idx, levels, width, height, color)


render_frame = render_frame_numpy if np is not None else render_frame_python


def main(argv=None):
    parser = argparse.ArgumentParser(description="A spinning doughnut, drawn in text.")
    parser.add_argument("-W", "--width", type=int, help="columns to draw in (default: terminal width)")
    parser.add_argument("-H", "--height", type=int, help="rows to draw in (default: terminal height - 1)")
    parser.add_argument("--color", action="store_true", help="use 24-bit colour instead of ASCII shading")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="target frames per second (default: 30); 0 runs flat out with no auto-tuning")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (default: run until Ctrl-C)")
    parser.add_argument("--fill", type=float, default=0.95, help="how much of the screen the donut fills (0-1)")
    parser.add_argument("--quality", type=float, default=6.0,
                        help="most surface samples per character cell (default: 6); the auto-tuner thins "
                             "this down to %g per cell if needed to hold --fps" % MIN_QUALITY)
    parser.add_argument("--dtheta", type=float,
                        help="sampling step around the tube (overrides --quality, and pins the density)")
    parser.add_argument("--dphi", type=float,
                        help="sampling step around the hole (overrides --quality, and pins the density)")
    args = parser.parse_args(argv)

    out = sys.stdout
    interactive = out.isatty()
    term = shutil.get_terminal_size(fallback=(80, 24))
    width = args.width or (term.columns - 1 if interactive else 80)
    height = args.height or (term.lines - 1 if interactive else 22)
    width, height = max(width, 8), max(height, 8)

    pinned = args.dtheta is not None or args.dphi is not None
    d_theta, d_phi = args.dtheta, args.dphi
    if d_theta is None or d_phi is None:
        auto_theta, auto_phi = auto_steps(width * height, max(args.quality, MIN_QUALITY))
        d_theta = auto_theta if d_theta is None else d_theta
        d_phi = auto_phi if d_phi is None else d_phi
    grid = torus_grid(R1, R2, d_theta, d_phi)

    # Explicit steps mean the caller has pinned the density, so don't second-guess
    # it; likewise --fps 0 means "as fast as it goes".
    tuner = None if (pinned or args.fps <= 0) else QualityTuner(grid, width * height)
    points = tuner.points if tuner else torus_grid_flat(grid)

    # A surface point at radius R (at most R1 + R2) and depth K2 + z projects to
    # at most R / sqrt(K2^2 - R^2) = 0.75 here, at the near rim. Dividing by that
    # makes --fill mean what it says, and keeps the donut inside the frame for
    # every rotation (the widest orientation is measured against the height,
    # which is why height is doubled: cells are about twice as tall as wide).
    max_radius = (R1 + R2) / math.sqrt(K2 * K2 - (R1 + R2) ** 2)
    scale = min(width / 2.0, height) * args.fill / max_radius

    budget = 1.0 / args.fps if args.fps > 0 else 0.0
    next_frame = time.perf_counter() + budget
    a = b = 0.0
    frames = 0

    if interactive:
        out.write("\x1b[2J")   # clear once; after that we just home the cursor
        out.write("\x1b[?25l")  # hide the cursor
    try:
        while True:
            started = time.perf_counter()
            frame = render_frame(points, a, b, width, height, scale, args.color)
            out.write(("\x1b[H" if interactive else "") + frame + "\n")
            out.flush()
            drawn = time.perf_counter() - started

            frames += 1
            if args.frames and frames >= args.frames:
                break

            # The first frame carries one-off costs (clearing, warming up, building
            # the colour table), so leave it out of the measurement.
            if tuner is not None and frames > 1 and tuner.observe(drawn, budget):
                points = tuner.points

            if budget:
                # Keep to a fixed schedule rather than sleeping a whole budget each
                # time: a sleep that runs late then costs a single frame instead of
                # dragging the frame rate down for the rest of the run.  Falling
                # behind resyncs instead of catching up in a burst.
                next_frame += budget
                idle = next_frame - time.perf_counter()
                if idle > 0:
                    time.sleep(idle)
                else:
                    next_frame = time.perf_counter()
            a += 0.04  # a full spin takes a few hundred frames
            b += 0.02
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            out.write("\x1b[0m\x1b[?25h")  # reset colour, show the cursor
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
