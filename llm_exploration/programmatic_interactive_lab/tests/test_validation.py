from __future__ import annotations

from execution.validation import extract_policy_source, validate_policy_source

VALID_SOURCE = "def policy(observation, memory):\n    return 0\n"


def test_extract_strips_single_markdown_fence():
    raw = f"```python\n{VALID_SOURCE}```"
    assert extract_policy_source(raw) == VALID_SOURCE.strip()


def test_extract_passes_through_raw_source():
    assert extract_policy_source(VALID_SOURCE) == VALID_SOURCE.strip()


def test_validate_accepts_valid_policy():
    outcome = validate_policy_source(VALID_SOURCE)
    assert outcome.valid
    assert outcome.error is None


def test_validate_rejects_syntax_error():
    outcome = validate_policy_source("def policy(observation)\n    return 0\n")
    assert not outcome.valid
    assert "Syntax error" in outcome.error


def test_validate_rejects_imports():
    outcome = validate_policy_source("import os\ndef policy(observation, memory):\n    return 0\n")
    assert not outcome.valid
    assert "Imports are not allowed" in outcome.error


def test_validate_rejects_missing_entry_point():
    outcome = validate_policy_source("def not_policy(observation):\n    return 0\n")
    assert not outcome.valid
    assert "policy(observation)" in outcome.error


def test_validate_rejects_one_argument_policy():
    # The historically-tolerated def policy(observation): form -- still
    # runnable if it's already saved on an existing node (see
    # execution/worker.py's _accepts_memory), but no longer accepted for
    # newly generated/validated code, so an LLM can't quietly drop memory
    # and fall back to a global/function-attribute substitute instead.
    outcome = validate_policy_source("def policy(observation):\n    return 0\n")
    assert not outcome.valid
    assert "memory" in outcome.error


def test_validate_allows_numpy_and_math_globals():
    source = "def policy(observation, memory):\n    return int(np.sum(observation) % 4 + math.floor(0))\n"
    outcome = validate_policy_source(source)
    assert outcome.valid, outcome.error


def test_validate_allows_random_global():
    source = "def policy(observation, memory):\n    return random.randint(0, 3)\n"
    outcome = validate_policy_source(source)
    assert outcome.valid, outcome.error


def test_validate_allows_newly_whitelisted_builtins():
    source = (
        "def policy(observation, memory):\n"
        "    ok = hasattr(observation, 'shape') and getattr(observation, 'shape', None) is not None\n"
        "    q, r = divmod(len(list(reversed([1, 2, 3]))), 2)\n"
        "    return int(pow(2, q) + r + ord(chr(65)) - ord('A') + (1 if ok else 0))\n"
    )
    outcome = validate_policy_source(source)
    assert outcome.valid, outcome.error


def test_validate_allows_collections_itertools_and_heapq_globals():
    source = (
        "def policy(observation, memory):\n"
        "    q = collections.deque([1, 2, 3])\n"
        "    q.append(4)\n"
        "    counts = collections.Counter([1, 1, 2])\n"
        "    pairs = list(itertools.product([0, 1], repeat=2))\n"
        "    heap = []\n"
        "    heapq.heappush(heap, 1)\n"
        "    heapq.heappush(heap, 0)\n"
        "    return heapq.heappop(heap) + len(pairs) + counts[1] + q.popleft()\n"
    )
    outcome = validate_policy_source(source)
    assert outcome.valid, outcome.error


def test_validate_allows_deque_counter_defaultdict_unqualified():
    source = (
        "def policy(observation, memory):\n"
        "    q = deque([1, 2, 3])\n"
        "    q.append(4)\n"
        "    counts = Counter([1, 1, 2])\n"
        "    groups = defaultdict(list)\n"
        "    groups['a'].append(1)\n"
        "    return q.popleft() + counts[1] + len(groups['a'])\n"
    )
    outcome = validate_policy_source(source)
    assert outcome.valid, outcome.error
