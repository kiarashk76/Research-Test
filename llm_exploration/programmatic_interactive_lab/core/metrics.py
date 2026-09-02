"""Session-level performance curves: episode return vs. cumulative
environment steps, cumulative LLM tokens, and elapsed wall-clock time --
analogous to ``training/trainer.py``'s ``plot_training_metrics`` in the
parent repo (same three x-axes), computed independently here rather than
imported, since this package depends on nothing from ``training``.

One thing does differ from the trainer's version: there, every LLM call
happens inside a specific ``agent.select_action`` during one episode, so
token counts attribute cleanly to that episode. Here, LLM calls happen in
Prompt Studio, independent of any single episode. So "cumulative tokens as
of episode N" is computed by wall-clock order: every LLM call that
completed at or before an episode ended counts toward that point on the
curve -- i.e. "how many tokens had this session spent by the time this
episode finished," not "tokens spent during this episode."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.experience import ExperienceStore
from core.llm import LLMCallStore


@dataclass
class EpisodePoint:
    episode_id: int
    episode_index: int
    actor_type: str
    actor_id: Optional[str]
    episode_return: float
    cumulative_env_steps: int
    cumulative_prompt_tokens: int
    cumulative_completion_tokens: int
    wall_time_seconds: float


def _parse_timestamp(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _build_points(episodes: list, calls: list) -> list[EpisodePoint]:
    """Shared by :func:`compute_session_metrics` and
    :func:`compute_training_run_metrics` -- ``episodes``/``calls`` are
    already filtered to whatever scope the caller wants (a whole session,
    or one training run), sorted chronologically."""
    if not episodes:
        return []

    start_time = _parse_timestamp(episodes[0].started_at)
    points: list[EpisodePoint] = []
    cumulative_steps = 0
    cumulative_prompt = 0
    cumulative_completion = 0
    call_index = 0

    for episode in episodes:
        cumulative_steps += episode.num_steps
        episode_end = _parse_timestamp(episode.ended_at)

        while call_index < len(calls) and _parse_timestamp(calls[call_index].created_at) <= episode_end:
            usage = calls[call_index].token_usage or {}
            cumulative_prompt += usage.get("prompt", 0)
            cumulative_completion += usage.get("completion", 0)
            call_index += 1

        points.append(EpisodePoint(
            episode_id=episode.id,
            episode_index=episode.episode_index,
            actor_type=episode.actor_type,
            actor_id=episode.actor_id,
            episode_return=episode.total_reward,
            cumulative_env_steps=cumulative_steps,
            cumulative_prompt_tokens=cumulative_prompt,
            cumulative_completion_tokens=cumulative_completion,
            wall_time_seconds=(episode_end - start_time).total_seconds(),
        ))

    return points


def compute_session_metrics(experience: ExperienceStore, llm_calls: LLMCallStore,
                             session_id: str) -> list[EpisodePoint]:
    """Chronological per-episode points for the session's performance curves.
    Only finished episodes (``ended_at`` set) are included."""
    episodes = sorted((e for e in experience.list_episodes() if e.ended_at is not None),
                       key=lambda e: e.started_at)
    calls = sorted(llm_calls.list(session_id), key=lambda c: c.created_at)
    return _build_points(episodes, calls)


def compute_training_run_metrics(context, train_run_id: str) -> list[EpisodePoint]:
    """Same per-episode performance curve as :func:`compute_session_metrics`,
    but scoped to one training run -- only episodes whose Run belongs to
    ``train_run_id`` (via that Run's own ``train_run_id`` metadata, the
    same tagging convention ``core.training``/``core.mcts`` already use),
    and only LLM calls tagged with that same ``train_run_id``. Lets the
    Train page plot one run's own learning curve instead of the whole
    session's, and compare several runs side by side."""
    run_ids = {run.id for run in context.runs.list()
               if (run.metadata or {}).get("train_run_id") == train_run_id}
    episodes = sorted(
        (e for e in context.experience.list_episodes()
         if e.ended_at is not None and e.run_id in run_ids),
        key=lambda e: e.started_at)
    calls = sorted(
        (c for c in context.llm_calls.list(context.session.id)
         if (c.metadata or {}).get("train_run_id") == train_run_id),
        key=lambda c: c.created_at)
    return _build_points(episodes, calls)


def _interp(x: float, curve: list[list[float]]) -> float:
    """Linear interpolation of ``curve`` (a list of ``[x, y]`` pairs sorted
    by ``x`` ascending -- true for every cumulative x-axis this module
    produces, since each only ever increases or holds steady over time) at
    ``x``. Clamps to the curve's own endpoints outside its range."""
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for i in range(1, len(curve)):
        x1, y1 = curve[i]
        if x1 >= x:
            x0, y0 = curve[i - 1]
            if x1 == x0:
                return y1
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return curve[-1][1]  # pragma: no cover -- unreachable given the guards above


def average_curves(curves: list[list[list[float]]], num_points: int = 50) -> list[list[float]]:
    """Combines several independent runs of "the same experiment" (same
    config, different random seeds) into one averaged line, for the Train
    page's "Compare training runs" section: resamples each ``[x, y]`` curve
    onto a shared grid spanning the x-range every curve covers (the latest
    of their starts to the earliest of their ends -- never extrapolating
    past a shorter curve's own data), linearly interpolating each at every
    grid point, then averaging the interpolated y-values across curves.

    A single curve (nothing to average against) is returned unchanged. If
    the curves' ranges don't overlap at all, falls back to the single
    longest curve rather than silently returning nothing.

    A thin ``[x, y]``-only wrapper around :func:`average_curves_with_band`,
    so every caller that doesn't need the variance band (Train/Plots pages)
    keeps working unchanged."""
    return [[x, y] for x, y, _std in average_curves_with_band(curves, num_points)]


def average_curves_with_band(curves: list[list[list[float]]], num_points: int = 50) -> list[list[float]]:
    """Same resampling grid and mean line as :func:`average_curves` (see its
    docstring), except each point is ``[x, mean, std]`` instead of ``[x,
    mean]`` -- ``std`` is the sample standard deviation (``ddof=1``) across
    curves at that grid point, i.e. the spread *across independent runs*
    (seeds), not within any one run. ``0.0`` when there's only one curve --
    nothing to compute a spread from.

    Kept as the one place this resampling happens (``average_curves`` is a
    wrapper around this, not a separate implementation) so a variance
    ribbon drawn around a group's averaged line (see
    ``ui/pages/evaluations.py``) is always computed from *exactly* the same
    grid points as the mean line itself -- never a second, independently
    resampled curve that could drift from it.
    """
    curves = [c for c in curves if c]
    if not curves:
        return []
    if len(curves) == 1:
        return [[x, y, 0.0] for x, y in curves[0]]

    start = max(c[0][0] for c in curves)
    end = min(c[-1][0] for c in curves)
    if start >= end:
        return [[x, y, 0.0] for x, y in max(curves, key=len)]

    step_count = max(num_points - 1, 1)
    xs_grid = [start + (end - start) * i / step_count for i in range(num_points)]
    n = len(curves)
    out = []
    for x in xs_grid:
        values = [_interp(x, curve) for curve in curves]
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        out.append([x, mean, variance ** 0.5])
    return out


def smooth_curve(points: list[list[float]], smoothing: float) -> list[list[float]]:
    """TensorBoard-style exponential moving average:
    ``smoothed[i] = smoothing * smoothed[i-1] + (1 - smoothing) * y[i]``.
    ``smoothing <= 0`` returns ``points`` unchanged; closer to (but below) 1
    smooths more heavily (more lag behind the raw values)."""
    if smoothing <= 0 or len(points) < 2:
        return points
    smoothed: list[list[float]] = []
    last: Optional[float] = None
    for x, y in points:
        last = y if last is None else smoothing * last + (1 - smoothing) * y
        smoothed.append([x, last])
    return smoothed
