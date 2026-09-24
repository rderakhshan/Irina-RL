"""Day 3 — skills: instructions the agent loads only when they are relevant.

A system prompt is expensive: every line is paid for on every turn of every
run. Most specialist knowledge — the house writing voice, how this project does
migrations, the release checklist — is needed on one task in twenty.

A skill splits that in two. The catalogue is one line per skill and always
present; the body is a whole document and is loaded only when the agent decides
it applies. The model does the routing, because the model is the only part of
the system that knows what the current task actually needs.

Design rules this file embodies:
  - A skill is a folder with a SKILL.md in it. No registry, no install step, no
    format to learn — a project author can add one with a text editor.
  - The description is the routing decision. It is the only thing the model
    sees before choosing, so it says when to use the skill, not what it is.
  - A miss is a helpful message, not an exception: the model misremembers a
    name, reads the list, and asks again.
"""

import os

SKILLS_DIR = "skills"
SKILL_FILE = "SKILL.md"


def catalog(workdir):
    """Return {name: {"description", "path"}} for every skill in the project."""
    root = os.path.join(os.path.realpath(workdir), SKILLS_DIR)
    if not os.path.isdir(root):
        return {}
    found = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, SKILL_FILE)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        found[name] = {"description": _description(text), "path": path}
    return found


def catalog_prompt(workdir):
    """Render the catalogue for the system prompt, or "" if there are none.

    Empty string rather than "no skills available": a sentence saying nothing
    is available still costs tokens on every turn and teaches the model
    nothing.
    """
    found = catalog(workdir)
    if not found:
        return ""
    lines = ["Skills available (load one with the use_skill tool when relevant):"]
    lines += [f"- {name}: {skill['description']}" for name, skill in found.items()]
    return "\n".join(lines)


def read_skill(workdir, name):
    """Return the full text of a skill, or a message naming the ones that exist."""
    found = catalog(workdir)
    skill = found.get(name)
    if skill is None:
        available = ", ".join(found) or "none"
        return f"ERROR: no skill named {name}. Available: {available}"
    with open(skill["path"], encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _description(text):
    """Pull `description:` out of YAML front matter, if the file has any.

    Front matter is optional and deliberately barely parsed — this is not YAML
    support, it is one convention that lets a skill carry its own summary.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.lower().startswith("description:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""
