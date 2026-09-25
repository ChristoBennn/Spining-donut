# spin

A spinning doughnut drawn in text, after Andy Sloane's 2006 [donut.c](https://www.a1k0n.net/2011/07/20/donut-math.html).

The shape is a genuine 3D torus, not a fake: each frame takes a fixed set of
surface points on the torus (and the outward normal at each point), rotates them
around the X axis and then the Z axis, perspective-projects them onto a grid of
character cells, lights each point by dotting its rotated normal with a fixed
light direction, and keeps only the nearest surface per cell with a z-buffer so
the donut looks solid instead of transparent.

## Running it

```sh
python3 donut.py              # classic ASCII, sized to your terminal
python3 donut.py --color      # 24-bit colour
python3 donut.py --fps 60     # aim higher; the tuner thins the sampling to suit
python3 donut.py -h           # all the knobs
```

No dependencies are required — it works in plain Python. If `numpy` is
installed, the same maths runs vectorised, which is much faster.

## Notes

The number of samples starts from the terminal size and is then adjusted every
frame to hold `--fps`, so an 80-column window and a 400-column one both stay
smooth. With `numpy` absent, an auto-tuner thins the sampling to keep up.
