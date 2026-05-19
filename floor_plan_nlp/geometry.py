# geometry.py

import math
import re

# ===========================================================================
# Path command parser
# ===========================================================================


def _tokenise_path(d: str):
    """Tokenise an SVG path ``d`` string into (command, [args]) tuples."""
    d = d.strip()
    tokens = re.findall(
        r"[MmLlHhVvAaQqTtCcSsZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d)
    cmds = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            i += 1
            nums = []
            while i < len(tokens) and not tokens[i].isalpha():
                nums.append(float(tokens[i]))
                i += 1
            cmds.append((cmd, nums))
        else:
            i += 1
    return cmds


def segment_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _arc_approx_length(x1, y1, rx, ry, x2, y2):
    """Approximate arc length using the average-radii heuristic."""
    chord = segment_length(x1, y1, x2, y2)
    avg_r = (abs(rx) + abs(ry)) / 2.0
    if avg_r < 1e-9:
        return chord
    # Angle subtended (very rough): chord ≈ 2r sin(θ/2)
    half_angle = math.asin(min(chord / (2 * avg_r), 1.0))
    return 2 * avg_r * half_angle  # arc length = r * θ


def parse_path_geometry(d: str):
    """
    Extract geometric info from an SVG path ``d`` attribute.

    Returns a list of segment dicts, each with:
        type        : "line" | "arc" | "curve" | "close"
        start       : (x, y)
        end         : (x, y)
        length      : float
        extra       : additional params (arc radii, etc.)
    """
    if not d:
        return []

    cmds = _tokenise_path(d)
    segments = []
    cx, cy = 0.0, 0.0   # current point
    start_x, start_y = 0.0, 0.0  # start of current sub-path

    for cmd, nums in cmds:
        if cmd == 'M':           # absolute move-to
            cx, cy = nums[0], nums[1]
            start_x, start_y = cx, cy
            # Implicit LineTo for additional pairs
            for k in range(2, len(nums), 2):
                nx, ny = nums[k], nums[k + 1]
                segments.append(dict(type="line", start=(cx, cy), end=(nx, ny),
                                     length=segment_length(cx, cy, nx, ny)))
                cx, cy = nx, ny

        elif cmd == 'm':         # relative move-to
            cx, cy = cx + nums[0], cy + nums[1]
            start_x, start_y = cx, cy

        elif cmd == 'L':         # absolute line-to
            for k in range(0, len(nums), 2):
                nx, ny = nums[k], nums[k + 1]
                segments.append(dict(type="line", start=(cx, cy), end=(nx, ny),
                                     length=segment_length(cx, cy, nx, ny)))
                cx, cy = nx, ny

        elif cmd == 'l':         # relative line-to
            for k in range(0, len(nums), 2):
                nx, ny = cx + nums[k], cy + nums[k + 1]
                segments.append(dict(type="line", start=(cx, cy), end=(nx, ny),
                                     length=segment_length(cx, cy, nx, ny)))
                cx, cy = nx, ny

        elif cmd == 'H':         # absolute horizontal line
            for x in nums:
                segments.append(dict(type="line", start=(cx, cy), end=(x, cy),
                                     length=abs(x - cx)))
                cx = x

        elif cmd == 'h':
            for dx in nums:
                segments.append(dict(type="line", start=(cx, cy), end=(cx + dx, cy),
                                     length=abs(dx)))
                cx += dx

        elif cmd == 'V':         # absolute vertical line
            for y in nums:
                segments.append(dict(type="line", start=(cx, cy), end=(cx, y),
                                     length=abs(y - cy)))
                cy = y

        elif cmd == 'v':
            for dy in nums:
                segments.append(dict(type="line", start=(cx, cy), end=(cx, cy + dy),
                                     length=abs(dy)))
                cy += dy

        elif cmd == 'A':         # absolute arc
            # params: rx ry x-rotation large-arc-flag sweep-flag x y
            for k in range(0, len(nums), 7):
                rx, ry = nums[k], nums[k + 1]
                nx, ny = nums[k + 5], nums[k + 6]
                length = _arc_approx_length(cx, cy, rx, ry, nx, ny)
                segments.append(dict(type="arc", start=(cx, cy), end=(nx, ny),
                                     length=length,
                                     extra={"rx": rx, "ry": ry,
                                            "x_rotation": nums[k + 2],
                                            "large_arc": nums[k + 3],
                                            "sweep": nums[k + 4]}))
                cx, cy = nx, ny

        elif cmd == 'a':         # relative arc
            for k in range(0, len(nums), 7):
                rx, ry = nums[k], nums[k + 1]
                nx, ny = cx + nums[k + 5], cy + nums[k + 6]
                length = _arc_approx_length(cx, cy, rx, ry, nx, ny)
                segments.append(dict(type="arc", start=(cx, cy), end=(nx, ny),
                                     length=length,
                                     extra={"rx": rx, "ry": ry,
                                            "x_rotation": nums[k + 2],
                                            "large_arc": nums[k + 3],
                                            "sweep": nums[k + 4]}))
                cx, cy = nx, ny

        elif cmd in ('C', 'c', 'Q', 'q', 'S', 's', 'T', 't'):
            # Cubic / quadratic Bézier – approximate as straight chord
            if cmd in ('C', 'S') or (cmd.lower() == 'c' and len(nums) >= 6):
                step = 6 if cmd in ('C', 'c') else 4
            else:
                step = 4
            step = max(step, 2)
            for k in range(0, len(nums) - 1, step):
                if k + 1 < len(nums):
                    if cmd.isupper():
                        nx, ny = nums[k + step - 2], nums[k + step - 1]
                    else:
                        nx, ny = cx + nums[k + step -
                                           2], cy + nums[k + step - 1]
                    segments.append(dict(type="curve", start=(cx, cy), end=(nx, ny),
                                         length=segment_length(cx, cy, nx, ny)))
                    cx, cy = nx, ny

        elif cmd in ('Z', 'z'):  # close path
            if (cx, cy) != (start_x, start_y):
                segments.append(dict(type="close", start=(cx, cy),
                                     end=(start_x, start_y),
                                     length=segment_length(cx, cy, start_x, start_y)))
            cx, cy = start_x, start_y

    return segments

# ===========================================================================
# Bounding box helpers
# ===========================================================================


def segments_bbox(segments):
    """Compute (min_x, min_y, max_x, max_y) from a list of segments."""
    xs, ys = [], []
    for seg in segments:
        xs += [seg["start"][0], seg["end"][0]]
        ys += [seg["start"][1], seg["end"][1]]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def total_length(segments):
    return sum(s["length"] for s in segments)
