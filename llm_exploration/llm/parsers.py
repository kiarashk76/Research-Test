"""Parse code out of an LLM response.

Format-level only - no domain knowledge. The LLM usually wraps code in Markdown
fences (```python ... ```); this strips them and returns runnable source.
"""

from __future__ import annotations

import re

# A Python fenced block: ```python\n <body> ```  (body captured, non-greedy).
_PY_FENCE = re.compile(r"```[ \t]*(?:python|py)[ \t]*\r?\n(.*?)```",
                       re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Return runnable Python source from an LLM response.

    * Extracts ```python``` / ```py``` fenced blocks only.
    * If there are **multiple** blocks, they are concatenated (in order) into
      one source string - so helper functions and the entry point all get
      defined together when you ``exec`` the result.
    * **Fails loudly** if no ```python``` block is present.

    Args:
        text: the raw LLM reply.

    Returns:
        The extracted source, stripped of surrounding whitespace.

    Raises:
        ValueError: if the response contains no ```python``` code block.
    """
    blocks = _PY_FENCE.findall(text)
    if not blocks:
        raise ValueError(
            "No ```python``` code block found in the LLM response; expected the "
            "code wrapped in a ```python ... ``` block. Got:\n"
            + (text[:500] + "..." if len(text) > 500 else text))
    return "\n\n".join(block.strip("\r\n") for block in blocks).strip()
