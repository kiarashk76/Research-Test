"""Shared plumbing for this repo's MiniHack task wrappers (see
``minihack_room_env.py`` and ``minihack_corridor_env.py``).

Every registered ``MiniHack-*`` navigation task returns the same
observation keys, decoded from the same raw NLE array shapes/dtypes
(``chars``/``message``/``blstats``/``screen_descriptions``, plus
``inv_strs``/``inv_letters`` for the skill-acquisition families that need
inventory visibility -- see ``SKILL_OBSERVATION_KEYS``) -- the
byte-decoding, the accurate ``gymnasium.spaces.Text`` construction, and
the action-labeling only need to exist once, not duplicated per task
family.
"""

from __future__ import annotations

import numpy as np

INSTALL_NOTES = (
    "MiniHack environments need the optional 'minihack' and 'nle' packages "
    "(nle compiles NetHack from source -- needs cmake>=3.28 and a C/C++ "
    "compiler; on macOS: `brew install cmake`, Xcode Command Line Tools). "
    "Install with: pip install minihack"
)

# What every wrapper here asks MiniHack/NLE to include in each observation --
# deliberately a small, LLM-relevant slice of everything NLE can return
# (glyphs/colors/specials/inventory/tty_* are all omitted): "chars" is the
# literal ASCII map (decoded to text below), "message" is the game's top
# status line, "blstats" is the numeric player-status vector (position,
# HP, etc. -- see NetHack Learning Environment's docs for the exact index
# meanings), "screen_descriptions" is NLE's own built-in per-cell plain-English
# description of what each character actually is (decoded to a compact
# (x, y, description) list below) -- already computed internally by NLE for
# every registered "MiniHack-*"/"NetHackScore-*" id, no extra package needed.
DEFAULT_OBSERVATION_KEYS = ("chars", "message", "blstats", "screen_descriptions")

# Same as DEFAULT_OBSERVATION_KEYS, plus "inv_strs"/"inv_letters" -- the
# skill-acquisition task wrappers (Simple Skills, LavaCross, WoD, Quest)
# routinely require wielding/wearing/eating/zapping/quaffing a *specific*
# carried item, so the agent needs to see what it's carrying; the plain
# navigation families above never need this (nothing in Room/Corridor/
# River/etc. depends on inventory contents), so it isn't part of the
# shared default.
SKILL_OBSERVATION_KEYS = DEFAULT_OBSERVATION_KEYS + ("inv_strs", "inv_letters")

# Appended to any MiniHack wrapper's ACTION_SPACE_DESCRIPTION whose action
# list includes at least one non-movement command -- every family here that
# has one (Corridor's open/kick, KeyRoom's apply, Simple Skills/LavaCross/
# WoD/Quest's much larger command sets) can trigger this same interaction
# pattern; Room/River/MazeWalk/HideNSeek never need it since their action
# lists are plain movement only, nothing that could ever open a follow-up
# prompt in the first place. The action list alone (names + keys, e.g.
# "eat (e)", "open (o)") discloses *what each action is called*, but not
# *how the interaction protocol actually works* -- that some commands open a
# follow-up prompt instead of completing immediately, sometimes needing a
# *direction* rather than a letter, that choosing an item from a list is a
# two-step lookup (find the item's letter in the current inventory, only
# then match that letter to an action's key) rather than something the
# prompt itself hands you directly, that keys mean something different
# depending on whether such a prompt is open, and that one kind of prompt
# can never be answered at all by this action space. Confirmed directly
# against live episodes, not assumed: pressing "eat (e)" while standing on
# food can trigger "There is an apple here; eat it? [ynq] (n)", answered not
# with any literal "yes" action -- there isn't one -- but with whichever
# action's key matches the letter shown, here "northwest (y)"; picking up an
# item and then wielding/wearing/zapping/eating it similarly opens an
# inventory-letter menu (e.g. "Eat what? [ade or ?*]") whose correct answer
# has to be read off the current `inventory` observation's own (letter,
# description) pairs, not derived by any letter arithmetic; separately,
# "open"/"kick" (Corridor and any other family
# whose action list includes them) trigger "In what direction?", answered by
# a normal compass-direction action (e.g. "open" then "east" opens the door
# to the east) -- no key-matching trick needed there, just a second,
# ordinary movement-labeled action; and separately again, "engrave" (and
# real NetHack's item-/character-naming prompts, though not exposed as
# actions here) asks for a typed line of free text -- confirmed unanswerable
# by this action space: the tool-selection sub-prompt it opens with already
# offers a choice ('-', "write with your fingers") that has no corresponding
# action at all, and there is no Enter/newline action either to submit a
# line even if one could be typed one character at a time. Without stating
# all of this explicitly, a model has no way to discover it from reward
# alone -- a wrong guess and the exact right answer often look identical
# (same 0 reward) until the correct follow-up is found, and a free-text
# prompt in particular looks exactly like a stuck/ignored action rather than
# an unanswerable one.
MINIHACK_ACTION_PROTOCOL_NOTE = (
    "In the action list, the letter or symbol shown in parentheses after each "
    "action's name (e.g. \"eat (e)\", \"northwest (y)\") is the single physical "
    "key that action presses -- not a label or shorthand, the literal key.\n"
    "Some actions do not complete immediately. Instead, they open a follow-up "
    "prompt that must be answered with your *next* action. There are four "
    "kinds of prompt:\n"
    "1. A yes/no question. Answer it with whichever action has '(y)' or '(n)' "
    "after its name -- even though that action's own name has nothing to do "
    "with 'yes'/'no', there is no action literally named that. The same key "
    "means a different thing depending on whether a prompt is currently open, "
    "so this rule only applies while one is open.\n"
    "2. A request to choose one item from a list, e.g. \"Eat what?\" or "
    "\"What do you want to wear?\". This is a two-step lookup, not a single "
    "letter shown by the prompt itself: first find the letter of the specific "
    "item you want in the current inventory (each entry is a (letter, "
    "description) pair, e.g. ('c', 'an apple') means that item's letter is "
    "'c'); then choose the action whose key (the letter in parentheses after "
    "its name) is exactly that same letter, e.g. the item lettered 'c' is "
    "selected by the action with '(c)' after its name. Do not compute this "
    "letter arithmetically (e.g. from its position in the alphabet) -- read "
    "it directly from the inventory entry and match it to an action's actual "
    "key.\n"
    "3. A request for a direction, e.g. \"In what direction?\" (from actions "
    "like \"open\"/\"kick\"). Answer with a normal compass-direction action -- "
    "e.g. after \"open\", the action \"east\" opens whatever is immediately "
    "east of you. This is an ordinary movement action, not a key-matching "
    "trick.\n"
    "4. A request to type free text, e.g. engraving a message or naming an "
    "item. This action space cannot type arbitrary text or submit a line, so "
    "this kind of prompt can never be completed. If one appears, answer with "
    "the \"esc\" action to cancel it, and prefer not to choose an action that "
    "opens this kind of prompt in the first place."
)

# Index of the first 5 blstats fields NLE documents as (x, y, strength,
# strength-hidden, dexterity, ...) -- only x/y are surfaced here (see
# format_blstats) since none of these navigation tasks otherwise vary
# player stats.
_BLSTATS_X, _BLSTATS_Y = 0, 1


def decode_chars(chars: np.ndarray) -> list:
    """NLE's "chars" observation is a 2D array of ASCII byte codes -- this
    is literally the map as it would render on screen. Decoded here (once,
    at the adapter boundary) into a **list of row strings** -- deliberately
    NOT a single string with embedded ``\\n`` separators (an earlier
    version of this function did exactly that), because a policy almost
    always needs to index into the map by (row, column), and
    ``for y, row in enumerate(chars)``/``chars[y][x]`` is the natural way
    to write that -- which only works if ``chars`` really is a list.
    Confirmed directly against real generated policies (not a
    hypothetical): with the old single-string form, that exact code
    pattern silently iterated one *character* at a time instead of one row
    (``row`` became a single character, the loop var became a flat
    character offset instead of a row index), permanently breaking any
    navigation logic built on it -- e.g. a derived column was always 0,
    so east/west movement could never be chosen at all. Making ``chars`` a
    real list removes the whole failure mode: the natural way to write the
    code is now also the correct way, no caveat for a model to remember or
    ignore. See core.environment._format_value_for_llm for how a list of
    row strings actually renders in an LLM-facing prompt (one line per
    row, each prefixed with its index)."""
    return ["".join(chr(c) for c in row) for row in chars]


def decode_message(message: np.ndarray) -> str:
    """NLE's "message" observation is a fixed-length, null-padded byte
    buffer -- decode to a plain (possibly empty) string."""
    text = bytes(int(b) for b in message).split(b"\x00", 1)[0]
    return text.decode("ascii", errors="replace")


def format_blstats(blstats: np.ndarray) -> str:
    return f"x={int(blstats[_BLSTATS_X])}, y={int(blstats[_BLSTATS_Y])}"


_AMBIGUOUS_CHARACTER_MAX_POSITIONS = 8


def decode_screen_descriptions(chars: np.ndarray, screen_descriptions: np.ndarray) -> list:
    """NLE's "screen_descriptions" observation is a (rows, cols, 80) array
    of fixed-length, null-padded byte buffers -- one plain-English
    description per map cell (e.g. "water", "a boulder", "staircase down"),
    computed by NLE itself from NetHack's own object/terrain database.

    Reduced here to a **legend**, not a per-cell list: one ``(character,
    description)`` pair per distinct pairing actually present in ``chars``
    this step, each listed once, in first-appearance (row-major) order.
    Positions aren't repeated here in general -- "chars" already gives the
    full grid, so a policy that needs a character's position(s) scans
    "chars" for it directly; this field only needs to answer "what does
    this character mean". A per-cell ``(x, y, description)`` list was
    tried first and rejected: NetHack's screen has far more generic-
    terrain description variants than expected (lit/unlit room floor,
    lit/unlit corridor, doorway, wall, ... -- verified against ``defsyms``
    in the installed ``nle`` package's own ``drawing.c``), so a per-cell
    list was mostly hundreds of duplicate entries for open floor.
    Deduplicating into a legend sidesteps needing to know that list at
    all -- a character that only ever means boring floor still only
    produces one legend line.

    One real exception, confirmed directly against a live
    ``MiniHack-Corridor-*`` observation: an open doorway with nothing
    standing in it renders as the exact same character as ordinary room
    floor ('.'), yet NLE's own ``screen_descriptions`` still correctly
    calls it "doorway" for that one cell and "floor of a room" everywhere
    else -- so the very same character ends up with two entries in the
    plain legend above, with nothing distinguishing *which* '.' is
    actually the room's exit. A policy (or a person) reading only the
    deduplicated legend has no way to find the doorway at all short of
    scanning the whole room hoping to spot the boundary gap by eye. Rare,
    genuinely ambiguous cases like this -- a character actually meaning
    more than one distinct thing within this same frame -- get their
    exact cell position(s) appended, capped at
    ``_AMBIGUOUS_CHARACTER_MAX_POSITIONS`` occurrences so a *common*
    reused character (plain floor, generic wall) never regresses into the
    hundreds-of-duplicates problem this legend was built to avoid; only
    genuinely rare, spatially-specific meanings (a doorway, a trap sharing
    a floor tile's glyph, ...) are small enough in practice to qualify.

    Cells whose description is empty (NLE's own encoding for
    unseen/off-map cells -- the whole per-cell byte buffer is zeroed)
    are skipped: an empty description is not a real character meaning."""
    rows, cols = chars.shape
    order: list[tuple[str, str]] = []
    positions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    descriptions_by_character: dict[str, set[str]] = {}
    for y in range(rows):
        for x in range(cols):
            character = chr(int(chars[y, x]))
            desc_bytes = bytes(screen_descriptions[y, x]).split(b"\x00", 1)[0]
            description = desc_bytes.decode("ascii", errors="replace")
            if not description:
                continue
            key = (character, description)
            if key not in positions:
                positions[key] = []
                order.append(key)
            positions[key].append((x, y))
            descriptions_by_character.setdefault(character, set()).add(description)

    legend = []
    for character, description in order:
        cell_positions = positions[(character, description)]
        ambiguous = len(descriptions_by_character[character]) > 1
        if ambiguous and len(cell_positions) <= _AMBIGUOUS_CHARACTER_MAX_POSITIONS:
            coords = ", ".join(f"({px},{py})" for px, py in cell_positions)
            legend.append((character, f"{description} (at {coords})"))
        else:
            legend.append((character, description))
    return legend


def decode_inventory(inv_strs: np.ndarray, inv_letters: np.ndarray) -> list:
    """NLE's "inv_strs"/"inv_letters" observations are a pair of parallel
    arrays -- up to 55 inventory slots, each an (inventory letter, plain-
    English description) pair (e.g. ('a', 'a +1 club (weapon in hand)')).
    An empty/unused slot has letter byte 0 and an all-zero description --
    verified directly against an actual observation, not assumed. Reduced
    here to a compact list of (letter, description) pairs, one per
    actually-occupied slot, skipping empty ones -- same "don't repeat the
    game's own null-padding as data" convention as
    :func:`decode_screen_descriptions`."""
    items = []
    for letter_byte, desc_bytes in zip(inv_letters, inv_strs):
        letter = int(letter_byte)
        if letter == 0:
            continue
        description = bytes(int(b) for b in desc_bytes).split(b"\x00", 1)[0].decode(
            "ascii", errors="replace")
        if not description:
            continue
        items.append((chr(letter), description))
    return items


def wrap_minihack_obs(obs: dict) -> dict:
    """Replaces the raw NLE arrays for any of ``chars``/``message``/
    ``blstats``/``screen_descriptions``/``inv_strs``+``inv_letters``
    present in ``obs`` with their decoded plain-text form -- the last pair
    collapses into a single ``"inventory"`` key (see
    :func:`decode_inventory`), since they're only ever meaningful
    together."""
    wrapped = dict(obs)
    if "screen_descriptions" in wrapped:
        # Must read the still-raw ``chars`` array (pre-decode below) --
        # decode_screen_descriptions pairs each cell's character with its
        # description.
        wrapped["screen_descriptions"] = decode_screen_descriptions(
            obs["chars"], wrapped["screen_descriptions"])
    if "inv_strs" in wrapped and "inv_letters" in wrapped:
        wrapped["inventory"] = decode_inventory(wrapped.pop("inv_strs"), wrapped.pop("inv_letters"))
    if "chars" in wrapped:
        wrapped["chars"] = decode_chars(wrapped["chars"])
    if "message" in wrapped:
        wrapped["message"] = decode_message(wrapped["message"])
    if "blstats" in wrapped:
        wrapped["blstats"] = format_blstats(wrapped["blstats"])
    return wrapped


def build_text_observation_space(raw_observation_space):
    """A ``gymnasium.spaces.Dict`` of ``Text`` spaces describing what
    ``wrap_minihack_obs`` actually returns for ``raw_observation_space``
    (the underlying, un-wrapped NLE env's own ``observation_space``) --
    NOT a copy of ``raw_observation_space`` itself, which describes the
    raw byte/int arrays instead of the decoded strings callers get back.

    ``spaces.Text``'s default charset is only letters+digits, which
    rejects most of what these fields actually contain (".", "#", "@",
    " ", "\\n", "=", ",", "-", ...). Each field's charset below is scoped
    to exactly what its decoder can produce -- verified against an actual
    terminal-frame observation, where NLE zeroes its internal buffer and
    ``decode_chars`` (unlike ``decode_message``, which stops at the first
    \\x00) turns every one of those zero bytes into a literal "\\x00"
    character, not the empty string one might expect.
    """
    from gymnasium import spaces

    rows, cols = raw_observation_space["chars"].shape
    message_len = raw_observation_space["message"].shape[0]

    chars_charset = frozenset(chr(c) for c in range(256))
    # decode_message ascii-decodes with errors="replace" and always stops
    # at the first \x00, so: bytes 1-127 decode to themselves, bytes
    # 128-255 become U+FFFD, and \x00 itself never appears.
    message_charset = frozenset(chr(c) for c in range(1, 128)) | {"�"}
    # Formatted as "x=<int>, y=<int>" from int64 blstats fields.
    blstats_charset = frozenset("xy=, -0123456789")

    spaces_dict = {
        # A list of `rows` row strings (see decode_chars), not one joined
        # string -- so this is a Sequence of per-row Text spaces, each up
        # to `cols` characters, rather than a single Text of length
        # rows*cols(+separators).
        "chars": spaces.Sequence(spaces.Text(max_length=cols, charset=chars_charset)),
        # min_length=0 -- there's usually no status message at all.
        "message": spaces.Text(min_length=0, max_length=message_len, charset=message_charset),
        # 32 comfortably covers the worst case ("x=-2147483648, "
        # "y=-2147483648" is 28 chars).
        "blstats": spaces.Text(max_length=32, charset=blstats_charset),
    }
    if "screen_descriptions" in raw_observation_space.spaces:
        # NLE_SCREEN_DESCRIPTION_LENGTH (80) includes the null terminator --
        # verified against nle.nethack.NLE_SCREEN_DESCRIPTION_LENGTH.
        spaces_dict["screen_descriptions"] = spaces.Sequence(spaces.Tuple((
            spaces.Text(min_length=1, max_length=1, charset=chars_charset),
            spaces.Text(max_length=79),
        )))
    if "inv_strs" in raw_observation_space.spaces and "inv_letters" in raw_observation_space.spaces:
        # inv_strs's raw shape is (55, 80) -- 80 includes the null
        # terminator, same convention as screen_descriptions above,
        # verified directly against an actual observation.
        inv_str_len = raw_observation_space["inv_strs"].shape[1]
        spaces_dict["inventory"] = spaces.Sequence(spaces.Tuple((
            spaces.Text(min_length=1, max_length=1, charset=chars_charset),
            spaces.Text(max_length=inv_str_len - 1),
        )))
    return spaces.Dict(spaces_dict)


def describe_observation_space(raw_observation_space) -> str:
    """A parameterized, plain-English description of what
    :func:`wrap_minihack_obs` actually returns for ``raw_observation_space``
    (mirrors :func:`build_text_observation_space`, which builds the
    matching ``gymnasium.spaces.Dict`` -- keep both in sync). Meant to be
    assigned as ``self.observation_space_description_hint`` by each
    concrete wrapper's ``__init__`` (see ``core.environment.EnvironmentAdapter``'s
    hint precedence) so the LLM prompt states this task's actual map
    dimensions/message length in words, rather than needing the raw Gym
    space repr as a crutch for that."""
    rows, cols = raw_observation_space["chars"].shape
    message_len = raw_observation_space["message"].shape[0]
    has_screen_descriptions = "screen_descriptions" in raw_observation_space.spaces
    has_inventory = ("inv_strs" in raw_observation_space.spaces
                     and "inv_letters" in raw_observation_space.spaces)
    text = (
        "The observation is a dict with these plain-text fields:\n"
        f"'chars': the currently visible map, a list of {rows} row strings of "
        f"{cols} characters each -- `chars[y]` is row y (top to bottom, 0-indexed) "
        "and `chars[y][x]` is the character at row y, column x. Characters are "
        "whatever the game draws for that cell (terrain, objects, creatures, or a "
        "blank/background character).\n"
        f"'message': the game's current status/message line, plain text, up to "
        f"{message_len} characters (empty when there is no message this step).\n"
        "'blstats': the agent's current position, given as the literal text "
        "'x=<int>, y=<int>'."
    )
    if has_screen_descriptions:
        text += (
            "\n'screen_descriptions': a legend for 'chars' -- a list of (character, "
            "description) pairs, one per distinct character actually appearing in "
            "'chars' this step (each listed once), telling you exactly what that "
            "character means right now (e.g. ('.', 'floor of a room'), ('}', 'water'), "
            "('`', 'a boulder'), ('>', 'staircase down'), ('@', a monster's name)). It "
            "does not repeat positions -- to find where a character is, scan 'chars' "
            "for it directly. A character with no known meaning yet (never seen this "
            "episode) is simply absent from the legend.\n"
            "The one exception: if the exact same character genuinely means two "
            "different things this step (e.g. an open doorway with nothing standing "
            "in it renders identically to plain room floor, both as '.'), each "
            "meaning gets its own legend entry, and the rarer one's exact cell "
            "position(s) are appended in parentheses (e.g. ('.', 'doorway (at (12,4))') "
            "alongside a separate ('.', 'floor of a room') entry) -- scanning 'chars' "
            "for that character alone cannot tell those cases apart, so check the "
            "legend for an appended position whenever a character has more than one "
            "listed meaning."
        )
    else:
        text += (
            " Character meanings are not explained here and must be discovered through "
            "interaction."
        )
    if has_inventory:
        text += (
            "\n'inventory': the agent's current carried items -- a list of (inventory "
            "letter, description) pairs, one per occupied slot (e.g. ('a', 'a +1 club "
            "(weapon in hand)')). An empty inventory is an empty list."
        )
    return text


_COMPASS_LABELS = {
    "N": "north", "E": "east", "S": "south", "W": "west",
    "NE": "northeast", "NW": "northwest", "SE": "southeast", "SW": "southwest",
}


def action_label(action) -> str:
    """A short, human/LLM-readable label for one NetHack action enum
    member (e.g. ``CompassDirection.N`` -> "north (k)", ``Command.OPEN``
    -> "open (o)").

    The full NetHack action set (used by the skill-acquisition task
    wrappers -- Simple Skills/LavaCross/WoD/Quest -- unlike the plain
    navigation families, which only ever use ``CompassDirection``)
    includes a *second* enum, ``CompassDirectionLonger``, whose members
    share the exact same ``.name`` values ("N", "E", ...) as
    ``CompassDirection`` but are a genuinely different action (move
    repeatedly in that direction until something interesting happens,
    NetHack's capital-letter "run" movement) -- without disambiguating by
    enum class here, two different action indices would render as the
    literal same label ("north"), leaving no way for an LLM reading the
    action list to tell them apart. Verified directly against an actual
    ``MiniHack-Eat-v0`` action list, not assumed.

    Every action's ``.value`` is literally the raw keystroke byte NetHack
    sends to the game (e.g. ``Command.FIRE`` isn't a distinct "fire"
    input mode -- it *is* the physical key 'f'). Critically, NetHack
    reuses that same alphabet for context-dependent prompts: a
    ``[ynq]``-style question is answered with 'y'/'n'/'q' (which are
    also, unrelatedly, the vi movement keys for northwest/southeast/quiver
    in normal play), and an inventory-letter prompt ("what do you want to
    eat? [f or ?*]") is answered with whichever letter that item was
    assigned, which can coincide with any other command's own letter
    (picking up the only item in a fresh room typically assigns it 'f',
    the same byte as ``Command.FIRE``) -- confirmed directly by stepping
    a real ``MiniHack-Eat-v0`` episode through such a prompt, not assumed.
    A policy (or a person using the Play page) has no way to make this
    connection from a label like "fire" alone, since it never reveals
    which literal key that is -- so the actual character is appended in
    parentheses on every label, letting a prompt's own shown letter be
    matched directly against the action list instead of requiring NetHack
    keybinding trivia."""
    name = getattr(action, "name", str(action))
    label = _COMPASS_LABELS.get(name, name.lower())
    if type(action).__name__ == "CompassDirectionLonger":
        label = f"run {label}"
    value = getattr(action, "value", None)
    if isinstance(value, int) and 32 <= value <= 126:
        label = f"{label} ({chr(value)})"
    return label
