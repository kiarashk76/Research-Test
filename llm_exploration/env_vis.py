"""Save snapshots of an environment's reset state for a list of seeds.

Standalone script: pass an environment name, (optionally) overrides, and a
list of seeds, and it drops one snapshot pair per seed into
outputs/<env>/<env_config>/ - the same folder the experiment hierarchy in
main.py uses for that env/config. Re-running with the same env/overrides
adds/overwrites snapshots for those seeds in that folder instead of creating
a separate folder; different overrides get their own folder (created if it
doesn't exist yet).

Each snapshot is one plain, uncolored image - there's nothing in the
environment that inherently maps to a color, so nothing is invented:
- observation_snapshot: whatever the observation is.
- rendered_snapshot: whatever ``env.render()`` returns.

Neither is assumed to be a single flat numeric grid. ``save_value_snapshot``
recursively flattens dicts and non-numeric lists/tuples into named leaves
(dispatching purely on each leaf's Python type: str / numeric array-like /
anything else), then draws every leaf as its own axis within one combined
figure. New environments with differently-shaped observations or render()
outputs never require touching this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import make_env, ENV_CONFIGS
from main import make_config_folder_name, _safe_name


def _as_numeric_array(value):
    """Return a 1D/2D numeric ndarray for ``value``, or None if it isn't one."""
    try:
        array = np.asarray(value)
    except Exception:
        return None

    if array.dtype.kind not in "iufb" or array.ndim not in (1, 2):
        return None

    return array.reshape(1, -1) if array.ndim == 1 else array


def _flatten_leaves(value, label: str = "value") -> list[tuple[str, object]]:
    """Recursively flatten dicts/non-numeric lists into (label, leaf) pairs.

    A leaf is anything that isn't a dict or a non-numeric list/tuple: a str,
    a numeric array/list (1D or 2D), or any other object (drawn via str()).
    """
    if isinstance(value, dict):
        leaves = []
        for key, sub_value in value.items():
            leaves.extend(_flatten_leaves(sub_value, f"{label}.{key}"))
        return leaves

    if isinstance(value, (list, tuple)) and _as_numeric_array(value) is None:
        leaves = []
        for index, sub_value in enumerate(value):
            leaves.extend(_flatten_leaves(sub_value, f"{label}[{index}]"))
        return leaves

    return [(label, value)]


def _draw_array(ax, array, label: str) -> None:
    """Draw a 2D numeric array as a plain grid of its values (no color)."""
    rows, cols = array.shape

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")

    # Grid lines on cell boundaries (half-integer offsets), not through the
    # cell centers where the numbers are drawn.
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(rows):
        for c in range(cols):
            value = array[r, c].item()
            text = f"{value:g}" if isinstance(value, float) else str(value)
            ax.text(c, r, text, ha="center", va="center", fontsize=9)

    ax.set_title(label, fontsize=10)


def _draw_text(ax, text: str, label: str) -> None:
    """Draw arbitrary text verbatim, monospaced (no color encoding)."""
    ax.axis("off")
    ax.text(
        0.5, 0.5, text,
        ha="center", va="center",
        fontsize=11, family="monospace",
        transform=ax.transAxes,
    )
    ax.set_title(label, fontsize=10)


def _draw_leaf(ax, value, label: str) -> None:
    if isinstance(value, str):
        _draw_text(ax, value, label)
        return

    array = _as_numeric_array(value)
    if array is not None:
        _draw_array(ax, array, label)
        return

    _draw_text(ax, str(value), label)


def save_value_snapshot(value, title: str, path: Path) -> None:
    """Render an arbitrary observation/render value as a single image.

    Dispatch is purely on Python type, never on which environment produced
    the value. Dicts and non-numeric lists/tuples are flattened (recursively)
    into named leaves, and every leaf gets its own axis within one figure.
    """
    leaves = _flatten_leaves(value)

    cols = min(3, len(leaves))
    rows = math.ceil(len(leaves) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, (label, leaf_value) in zip(flat_axes, leaves):
        _draw_leaf(ax, leaf_value, label)

    for ax in flat_axes[len(leaves):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save arbitrary snapshots of an environment."
    )

    parser.add_argument("--env", default="SimpleGridEnv")
    parser.add_argument("--env-overrides", default="{}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 45, 68])

    return parser.parse_args()


def main():
    args = parse_args()
    overrides = json.loads(args.env_overrides)

    env, resolved_params = make_env(args.env, overrides=overrides)
    env_name = type(env).__name__
    config_name = make_config_folder_name(ENV_CONFIGS[env_name], resolved_params)

    output_dir = (
        Path(__file__).resolve().parent
        / "outputs"
        / _safe_name(env_name)
        / config_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        obs, _ = env.reset(seed=seed)
        render_output = env.render()

        title = f"{env_name} ({config_name}) - seed {seed}"

        obs_path = output_dir / f"observation_snapshot_seed{seed}.png"
        save_value_snapshot(obs, title, obs_path)
        print(f"Saved {obs_path}")

        render_path = output_dir / f"rendered_snapshot_seed{seed}.png"
        save_value_snapshot(render_output, title, render_path)
        print(f"Saved {render_path}")

    print(f"\nSaved {len(args.seeds)} snapshot pair(s) to: {output_dir}")


if __name__ == "__main__":
    main()
