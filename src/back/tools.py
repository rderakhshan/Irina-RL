"""Day 2 — the tools: the harness's hands, and the schemas that describe them.

Yesterday a tool was a hand-written class. That does not scale: the schema and
the function drift apart the moment someone edits one and forgets the other.
Today the schema is *derived* from the function, so the two cannot disagree.

The six tools below are the whole job. Read, write, edit, run, list, search is
what a coding agent does; everything else is a convenience built on top.

Design rules this file embodies:
  - One gate, not six. Every path in this module goes through `resolve`, so
    "stay inside the working directory" is enforced in exactly one place.
  - Failures are text, not exceptions, whenever the model could fix them. A
    missing snippet comes back as advice; escaping the sandbox does not.
  - Every output is bounded. A tool result is spent context, so reads, listings
    and command output all truncate at a size a turn can afford.
  - Every parameter is a string. Models emit JSON loosely; the tool coerces at
    the boundary rather than trusting a type the schema merely asked for.
"""

import fnmatch
import inspect
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

MAX_READ_LINES = 4000     # a read past this is a search, not a read
MAX_OUTPUT_CHARS = 12000  # head and tail of a long command's output
MAX_LIST_ENTRIES = 500
MAX_GREP_HITS = 200
MAX_LINE_CHARS = 200      # a matched minified line must not eat the turn
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv"}


@dataclass
class Tool:
    """A tool in its entirety: a name, the schema the model reads, the Python
    the harness calls. The loop needs nothing else, which is why no framework
    is required to add one.
    """

    name: str
    spec: dict
    run: Callable


def tool(description, **params):
    """Turn a plain function into a `Tool`, deriving its schema from itself.

    `description` is what the model reads about the tool; each keyword argument
    documents the parameter of the same name. Arguments without a default are
    required, those with one are optional — the signature is the single source
    of truth, so a renamed argument cannot leave a stale schema behind.
    """
    def decorate(fn):
        args = inspect.signature(fn).parameters
        spec = {"schema": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                # Every property is a string on purpose: see the module header.
                "properties": {name: {"type": "string",
                                      "description": params.get(name, "")}
                               for name in args},
                "required": [name for name, arg in args.items()
                             if arg.default is inspect.Parameter.empty],
            },
        }}
        return Tool(name=fn.__name__, spec=spec, run=fn)
    return decorate


def core_tools(workdir):
    """Build the six core tools, each closed over the real path of `workdir`.

    Closing over the directory rather than passing it per call means the model
    never gets to name the sandbox, and therefore never gets to change it.
    """
    root = os.path.realpath(workdir)
    os.makedirs(root, exist_ok=True)

    def resolve(path):
        """Turn a model-supplied path into an absolute one inside `root`.

        `realpath` first, so a symlink pointing out of the sandbox is caught by
        where it lands rather than by how it is spelled. The prefix test adds a
        separator so that a sibling directory like `/tmp/work-evil` cannot pass
        as a child of `/tmp/work`.
        """
        full = os.path.realpath(os.path.join(root, path))
        if full != root and not full.startswith(root + os.sep):
            raise PermissionError(f"{path!r} escapes the working directory")
        return full

    @tool("Read a file from the working directory, with line numbers",
          path="Path to the file, relative to the working directory")
    def read_file(path):
        """Return the file numbered "N<TAB>line", truncated if it is huge."""
        with open(resolve(path), encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
        # Numbering is not decoration: it is how the model cites a location
        # back to you, and how it counts lines it is about to edit.
        body = "\n".join(f"{n}\t{line}"
                         for n, line in enumerate(lines[:MAX_READ_LINES], 1))
        if len(lines) > MAX_READ_LINES:
            body += (f"\n... truncated at {MAX_READ_LINES} lines; "
                     f"the file has {len(lines)} lines")
        return body

    @tool("Write a file, creating parent directories and overwriting any "
          "existing file",
          path="Path to write, relative to the working directory",
          content="The full contents of the file")
    def write_file(path, content):
        """Write `content` to `path` and report how much landed where."""
        full = resolve(path)
        os.makedirs(os.path.dirname(full) or root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(content)
        return f"Wrote {len(content)} chars to {path}"

    @tool("Replace an exact snippet in a file; the snippet must appear exactly "
          "once",
          path="Path to the file, relative to the working directory",
          old="The exact text to replace, copied from the file",
          new="The text to put in its place")
    def edit_file(path, old, new):
        """Replace `old` with `new`, but only when the target is unambiguous."""
        full = resolve(path)
        with open(full, encoding="utf-8") as handle:
            text = handle.read()
        found = text.count(old)
        # Both failures are recoverable, so they read as instructions: the
        # model's next turn is a better read or a longer snippet, not a stall.
        if found == 0:
            return "ERROR: snippet not found — read the file and copy it exactly"
        if found > 1:
            return (f"ERROR: snippet appears {found} times — include more "
                    f"context to make it unique")
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(text.replace(old, new, 1))
        return f"Edited {path}"

    @tool("Run a shell command in the working directory and return its output",
          command="The shell command to run",
          timeout="Seconds to wait before giving up (default 120)")
    def bash(command, timeout="120"):
        """Run one command and return its combined output, bounded and blocking."""
        seconds = int(float(str(timeout).strip()))
        try:
            done = subprocess.run(command, shell=True, cwd=root,
                                  capture_output=True, text=True,
                                  timeout=seconds)
        except subprocess.TimeoutExpired:
            # A hung command is an observation, not a crash: the model usually
            # retries with a narrower command or a longer timeout.
            return f"ERROR: timed out after {seconds}s"
        # stdout and stderr are merged because the model reads them the way a
        # human reads a terminal — interleaved, as one story.
        output = (done.stdout + done.stderr).strip()
        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            cut = len(output) - MAX_OUTPUT_CHARS
            output = (f"{output[:half]}\n"
                      f"... [{cut} chars truncated] ...\n{output[-half:]}")
        # Silence is ambiguous; the exit code says whether it was the good kind.
        return output or f"(exit {done.returncode}, no output)"

    @tool("List files in the working directory matching a glob pattern",
          pattern="Glob such as **/*.py, matched against path and filename")
    def list_files(pattern="**/*"):
        """Walk the tree and return matching paths, sorted and capped."""
        hits = []
        for folder, subfolders, filenames in os.walk(root):
            # Pruning in place stops os.walk descending into the noise at all.
            subfolders[:] = [d for d in subfolders if d not in IGNORE_DIRS]
            for name in filenames:
                rel = os.path.relpath(os.path.join(folder, name), root)
                # Matched against both, so "*.py" and "src/*.py" both behave
                # the way the person writing the pattern expected.
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                    hits.append(rel)
        hits.sort()
        body = "\n".join(hits[:MAX_LIST_ENTRIES]) or "(no matches)"
        if len(hits) > MAX_LIST_ENTRIES:
            body += f"\n... and {len(hits) - MAX_LIST_ENTRIES} more"
        return body

    @tool("Search file contents for a regular expression",
          regex="Python regular expression to search for",
          pattern="Glob limiting which files are searched (default all)")
    def grep(regex, pattern="*"):
        """Return "path:lineno: text" for each match, clipped and capped."""
        compiled = re.compile(regex)
        hits = []
        for folder, subfolders, filenames in os.walk(root):
            subfolders[:] = [d for d in subfolders if d not in IGNORE_DIRS]
            for name in sorted(filenames):
                rel = os.path.relpath(os.path.join(folder, name), root)
                if not (fnmatch.fnmatch(rel, pattern)
                        or fnmatch.fnmatch(name, pattern)):
                    continue
                try:
                    with open(os.path.join(folder, name), encoding="utf-8",
                              errors="replace") as handle:
                        lines = handle.read().splitlines()
                except OSError:
                    # An unreadable file is not worth failing a search over.
                    continue
                for number, line in enumerate(lines, 1):
                    if compiled.search(line):
                        hits.append(f"{rel}:{number}: {line[:MAX_LINE_CHARS]}")
                        if len(hits) >= MAX_GREP_HITS:
                            return "\n".join(hits) + f"\n... stopped at {MAX_GREP_HITS} hits"
        return "\n".join(hits) or "(no matches)"

    return [read_file, write_file, edit_file, bash, list_files, grep]
