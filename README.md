# spinning-donut

A spinning doughnut drawn entirely in text, after Andy Sloane's 2006
[donut.c](https://www.a1k0n.net/2011/07/20/donut-math.html).

The shape is a genuine 3D torus, not a fake: each frame takes a fixed set of
surface points on the torus (and the outward normal at each point), rotates them
around the X axis and then the Z axis, perspective-projects them onto a grid of
character cells, lights each point by dotting its rotated normal with a fixed
light direction, and keeps only the nearest surface per cell with a z-buffer so
the donut looks solid instead of transparent.

It runs in a terminal, sizes itself to your window, and can draw in plain ASCII
shading or in 24-bit colour.

## Requirements

- **Python 3.9 or newer.** Nothing else is required — no third-party packages.
- **numpy is optional.** If it is installed, the same maths runs vectorised,
  which is much faster. Without it the donut still spins; the auto-tuner just
  thins the sampling to keep the frame rate up.

Check your version with:

```sh
python3 --version     # or: python --version
```

Optionally, for the faster path:

```sh
pip install numpy      # or: conda install numpy
```

## Running it

From the folder containing `donut.py`:

```sh
python3 donut.py              # classic ASCII, sized to your terminal
python3 donut.py --color      # 24-bit colour
python3 donut.py --fps 60     # aim higher; the tuner thins the sampling to suit
python3 donut.py -h           # all the knobs
```

On Windows, `python donut.py` or `py donut.py` usually works instead.

Stop it with **Ctrl-C**. It spins until you interrupt it.

Run it in a **real terminal window**. The animation clears the screen and homes
the cursor, so if the output is captured rather than drawn to a terminal (a
notebook, an IDE output pane, a pipe) you will just see frame after frame instead
of a spinning donut.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `-W`, `--width N` | terminal width | Columns to draw in. |
| `-H`, `--height N` | terminal height − 1 | Rows to draw in. |
| `--color` | off | Use 24-bit colour instead of ASCII shading. |
| `--fps N` | `30` | Target frames per second. `0` runs flat out with no auto-tuning. |
| `--frames N` | `0` | Stop after N frames. `0` runs until Ctrl-C. |
| `--fill F` | `0.95` | How much of the screen the donut fills (`0`–`1`). |
| `--quality Q` | `6` | Most surface samples per character cell; the auto-tuner thins this down, never below 3 per cell. |
| `--dtheta D` | auto | Sampling step around the tube. Overrides `--quality` and pins the density. |
| `--dphi D` | auto | Sampling step around the hole. Overrides `--quality` and pins the density. |

### Examples

```sh
# A colour donut in a fixed 100x40 area
python3 donut.py --color -W 100 -H 40

# Denser sampling for a sharp, slow-ish render (no thinning)
python3 donut.py --quality 12 --fps 0

# Pin the sampling by hand
python3 donut.py --dtheta 0.05 --dphi 0.02

# Render 200 frames and quit — handy for piping to a file
python3 donut.py --frames 200 > frames.txt
```

## How it works

Per frame, for every sample point:

1. **Rotate** the point around the X axis by angle `a`, then around the Z axis by
   angle `b` (the two angles advance every frame, which is what makes it spin).
2. **Project** it onto a grid of character cells by dividing by the depth — that
   division is the perspective, and it is why the far side looks smaller.
3. **Light** it by dotting the rotated surface normal with a fixed light
   direction, then pick a character from a dim-to-bright ramp (`.` through `@`),
   or a colour from a gradient in `--color` mode.
4. **Resolve depth** with a z-buffer, keeping only the nearest surface per cell
   so the donut looks solid instead of transparent.

The sampling density starts from the terminal size and is then adjusted every
frame to hold `--fps`: thin sampling is the same thing a coarser sampling step
would have produced, so the tuner thins the precomputed grid by an integer stride
and the cost tracks the sample count. That way an 80-column window and a
400-column one both stay smooth. Below about three samples per cell the surface
starts to show gaps, so the tuner stops there.

With numpy the rotation, projection, lighting and z-buffer are all done on whole
arrays — the z-buffer is a scatter-max (`np.maximum.at`), one pass instead of a
Python loop. Without numpy the identical arithmetic runs in a plain Python loop,
and both engines produce the same picture character for character.

## Credits

The maths and the idea are Andy Sloane's, from his 2006 `donut.c`
([write-up](https://www.a1k0n.net/2011/07/20/donut-math.html)).
