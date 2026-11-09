"""Builds Instinct-format next-edit prompts.

The SAME function is used for live inference and for training, so whatever the
model learns from your feedback matches exactly what it sees at inference time.
Format follows continuedev/instinct (utils/datautils.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SYSTEM_PROMPT = """You are Instinct, an intelligent next-edit predictor developed by Continue.dev. Your role as an AI assistant is to help developers complete their code tasks by predicting the next edit that they will make within the section of code marked by <|editable_region_start|> and <|editable_region_end|> tags.

You have access to the following information to help you make informed suggestions:

- Context: In the section marked \"### Context\", there are context items from potentially relevant files in the developer's codebase. Each context item consists of a <|context_file|> marker, the filepath, a <|snippet|> marker, and then some content from that file, in that order. Keep in mind that not all of the context information may be relevant to the task, so use your judgement to determine which parts to consider.
- User Edits: In the section marked \"### User Edits:\", there is a record of the most recent changes made to the code, helping you understand the evolution of the code and the developer's intentions. These changes are listed from most recent to least recent. It's possible that some of the edit diff history is entirely irrelevant to the developer's change. The changes are provided in a unified line-diff format, i.e. with pluses and minuses for additions and deletions to the code.
- User Excerpt: In the section marked \"### User Excerpt:\", there is a filepath to the developer's current file, and then an excerpt from that file. The <|editable_region_start|> and <|editable_region_end|> markers are within this excerpt. Your job is to rewrite only this editable region, not the whole excerpt. The excerpt provides additional context on the surroundings of the developer's edit.
- Cursor Position: Within the user excerpt's editable region, the <|user_cursor_is_here|> flag indicates where the developer's cursor is currently located, which can be crucial for understanding what part of the code they are focusing on. Do not produce this marker in your output; simply take it into account.

Your task is to predict and complete the changes the developer would have made next in the editable region. The developer may have stopped in the middle of typing. Your goal is to keep the developer on the path that you think they're following. Some examples include further implementing a class, method, or variable, or improving the quality of the code. Make sure the developer doesn't get distracted by ensuring your suggestion is relevant. Consider what changes need to be made next, if any. If you think changes should be made, ask yourself if this is truly what needs to happen. If you are confident about it, then proceed with the changes.

# Steps

1. **Review Context**: Analyze the context from the resources provided, such as recently viewed snippets, edit history, surrounding code, and cursor location.
2. **Evaluate Current Code**: Determine if the current code within the tags requires any corrections or enhancements.
3. **Suggest Edits**: If changes are required, ensure they align with the developer's patterns and improve code quality.
4. **Maintain Consistency**: Ensure indentation and formatting follow the existing code style.

# Output Format

- Provide only the revised code within the tags. Do not include the tags in your output.
- Ensure that you do not output duplicate code that exists outside of these tags.
- Avoid undoing or reverting the developer's last change unless there are obvious typos or errors.

"""

PROMPT_PREFACE = (
    "Reference the user excerpt, user edits, and the snippets to understand the "
    "developer's intent. Update the editable region of the user excerpt by predicting "
    "and completing the changes they would have made next. This may be a deletion, "
    "addition, or modification of code."
)

EDIT_START = "<|editable_region_start|>"
EDIT_END = "<|editable_region_end|>"
CURSOR = "<|user_cursor_is_here|>"
MARKERS = (EDIT_START, EDIT_END, CURSOR)

EXCERPT_RADIUS = 25      # lines of excerpt above/below cursor
EDITABLE_RADIUS = 5      # lines of editable region above/below cursor
MAX_PROMPT_CHARS = 60_000


def _clean(text: str) -> str:
    for m in MARKERS:
        text = text.replace(m, "")
    return text


@dataclass
class BuiltPrompt:
    user_prompt: str
    region_start: int      # absolute 0-based line in the file (inclusive)
    region_end: int        # absolute 0-based line in the file (exclusive)
    original_region: str   # current text of the editable region


def build_prompt(
    filepath: str,
    lines: list[str],
    lines_start: int,
    cursor_line: int,
    cursor_char: int,
    recent_edits: list[dict] | None = None,
    context: list[dict] | None = None,
) -> BuiltPrompt:
    """lines: a window of the file, where lines[0] is file line `lines_start`.
    cursor_line is absolute (file) 0-based; recent_edits = [{filepath, diff}]
    newest first; context = [{filepath, content}]."""
    if not lines:
        lines = [""]
    lines = [_clean(l) for l in lines]
    c = cursor_line - lines_start
    c = max(0, min(c, len(lines) - 1))

    ex_start = max(0, c - EXCERPT_RADIUS)
    ex_end = min(len(lines), c + EXCERPT_RADIUS + 1)
    ex = lines[ex_start:ex_end]
    cl = c - ex_start

    ed_start = max(0, cl - EDITABLE_RADIUS)
    ed_end = min(len(ex), cl + EDITABLE_RADIUS + 1)

    cur = ex[cl]
    cc = max(0, min(cursor_char, len(cur)))
    rendered = (
        ex[:ed_start]
        + [EDIT_START]
        + ex[ed_start:cl]
        + [cur[:cc] + CURSOR + cur[cc:]]
        + ex[cl + 1:ed_end]
        + [EDIT_END]
        + ex[ed_end:]
    )

    ctx_parts = []
    for item in (context or [])[:6]:
        content = _clean(str(item.get("content", ""))).rstrip()
        if not content.strip():
            continue
        ctx_parts.append(f"<|context_file|> {item.get('filepath', 'unknown')}\n<|snippet|>\n{content}")
    context_str = "\n\n".join(ctx_parts)

    prompt = PROMPT_PREFACE + f"\n\n### Context:\n\n{context_str}\n\n"
    prompt += "\n### User Edits:\n\n"
    for e in (recent_edits or [])[:5]:
        diff = _clean(str(e.get("diff", ""))).strip("\n")
        if not diff:
            continue
        prompt += f'User edited file "{e.get("filepath", filepath)}"\n\n'
        prompt += "```diff\n" + diff + "\n```\n\n"
    prompt += "### User Excerpt:\n"
    prompt += '"' + filepath + '"\n\n'
    prompt += "\n".join(rendered)

    if len(prompt) > MAX_PROMPT_CHARS and ctx_parts:
        # drop context first if too long
        return build_prompt(filepath, lines, lines_start, cursor_line, cursor_char, recent_edits, [])

    abs_start = lines_start + ex_start + ed_start
    abs_end = lines_start + ex_start + ed_end
    return BuiltPrompt(prompt, abs_start, abs_end, "\n".join(ex[ed_start:ed_end]))


_FENCE = re.compile(r"^```[\w+-]*\n(.*?)\n?```\s*$", re.S)


def parse_completion(text: str) -> str:
    """Turn raw model output into the new editable-region text."""
    s = text
    i = s.rfind(EDIT_START)
    if i != -1:
        s = s[i + len(EDIT_START):]
        j = s.rfind(EDIT_END)
        if j != -1:
            s = s[:j]
    elif EDIT_END in s:
        s = s[: s.rfind(EDIT_END)]
    m = _FENCE.match(s.strip("\n"))
    if m:
        s = m.group(1)
    s = s.replace(CURSOR, "")
    return s.strip("\n")


def messages_for(user_prompt: str, target: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs


def same(a: str, b: str) -> bool:
    norm = lambda s: "\n".join(l.rstrip() for l in s.strip("\n").splitlines())
    return norm(a) == norm(b)
