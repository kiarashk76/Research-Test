"""Static validation/extraction for LLM-generated policy source.

Uses this package's own ``execution/sandbox.py`` (de-fencing, AST-based
import rejection, restricted-builtins ``exec``, and an entry-point check --
see that module's docstring for why it's a self-contained copy rather than
an import from ``agents``). Keeping validation logic identical to what
``execution/policy_runner.py``'s subprocess will actually accept means
"validated" and "runnable" never diverge -- with one deliberate exception:
:func:`validate_policy_source` additionally requires the two-argument
``def policy(observation, memory):`` form (see ``_requires_memory_parameter``
below), while ``execution/worker.py``'s ``_accepts_memory`` still happily
*runs* an older one-argument ``def policy(observation):`` unchanged. That
asymmetry is intentional, not a bug to fix: it lets every *already-saved*
node (whose code predates this requirement) keep replaying/re-training
exactly as before, while blocking any *newly generated* policy from quietly
reverting to the one-argument form -- see ``_requires_memory_parameter``'s
docstring for why that form is worth blocking at all.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

from execution.sandbox import compile_policy_source, strip_code_fences

ENTRY_POINT_HINT = "def policy(observation, memory): ... return action -- memory is required, not optional"

MISSING_MEMORY_PARAMETER_ERROR = (
    "policy(...) must accept two parameters: def policy(observation, memory): ... return action. "
    "A single-argument def policy(observation): is no longer accepted for newly generated code. "
    "Do not work around this with a global variable or a function attribute (e.g. policy.memory) "
    "as a substitute for the real memory argument -- that state would never reset between "
    "episodes the way the real memory argument does, and would be invisible to any future "
    "critique/repair step, since only the real memory argument is captured, shown as evidence, "
    "and reset to {} at the start of each new episode."
)


@dataclass
class ValidationOutcome:
    source: str
    valid: bool
    error: Optional[str]


def extract_policy_source(raw_response: str) -> str:
    """Pull the intended policy source out of a raw LLM response."""
    return strip_code_fences(raw_response)


def _requires_memory_parameter(policy_fn) -> Optional[str]:
    """``None`` if ``policy_fn`` declares a second (``memory``) parameter,
    otherwise :data:`MISSING_MEMORY_PARAMETER_ERROR`.

    A one-argument ``def policy(observation):`` is still accepted at
    *execution* time (see this module's own docstring), but newly generated
    code that omits ``memory`` has, in practice, reliably meant the policy
    reaches for a global variable or a function attribute (``policy.memory``)
    as its own private substitute instead -- observed directly in generated
    policies reviewed this session. Rejecting the one-argument form here,
    with an error fed back the same way any other invalid attempt is (see
    this repo's "even an invalid attempt is data" convention), stops that
    workaround from being generated in the first place rather than relying
    on prompt wording alone to discourage it."""
    try:
        param_count = len(inspect.signature(policy_fn).parameters)
    except (TypeError, ValueError):
        # A signature that can't be introspected (e.g. a builtin) -- can't
        # possibly declare a second parameter, so this fails closed.
        param_count = 0
    return None if param_count >= 2 else MISSING_MEMORY_PARAMETER_ERROR


def validate_policy_source(source: str) -> ValidationOutcome:
    """Syntax + no-imports + defines callable ``policy(observation, memory)``
    -- see module docstring for why this must match ``PolicyRunner``'s
    worker exactly, except for the one deliberate divergence documented
    there (a saved one-argument policy still *runs*; only newly validated
    code must declare ``memory``)."""
    policy_fn, error = compile_policy_source(source)
    if error is not None:
        return ValidationOutcome(source=source, valid=False, error=error)
    memory_error = _requires_memory_parameter(policy_fn)
    if memory_error is not None:
        return ValidationOutcome(source=source, valid=False, error=memory_error)
    return ValidationOutcome(source=source, valid=True, error=None)
