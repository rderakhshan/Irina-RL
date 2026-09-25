"""Chat history: the raw, append-only record of every session you've had.

This is the *raw material* for offline learning on your own chats — the
transcripts themselves, before any DeepSeek gate is applied. Two things write
here:

  * **`ask()`** — every pure Q&A ("Ask the student") is saved.
  * **`collect()`** — every collected session transcript is saved too.

Nothing here is ever graded, distilled, or altered at write time. The gate
(low-scoring sessions rejected, good ones distilled into goldens by DeepSeek)
is applied *later* — in batches, by the offline train-on-history button. That
way you pay DeepSeek at most once per transcript, and only when you actually
ask for offline learning.

Lines are JSONL. `processed` flips from false → true once a transcript has been
judged, so it is never graded (and never billed) twice.
"""

import json
import os
import threading


class ChatHistory:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()

    # -- write side -----------------------------------------------------

    def append(self, entry):
        """Persist one transcript line: `{"ts", "kind", "issue", "messages",
        "final", "processed": False}`. Messages is the full assistant/user
        transcript list (role + text), so DeepSeek can judge it later."""
        entry = dict(entry)
        entry.setdefault("processed", False)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- read side ------------------------------------------------------

    def entries(self):
        """All stored transcripts, in append order. Resilient to the file
        being half-written: a damaged line is skipped, not fatal."""
        if not os.path.exists(self.path):
            return []
        out = []
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        return out

    def pending(self):
        """Transcripts that have not yet been graded+distilled."""
        return [e for e in self.entries() if not e.get("processed")]

    def stats(self):
        es = self.entries()
        words = 0
        for e in es:
            for m in e.get("messages", []):
                words += len(m.get("text", "").split())
        return {
            "all": len(es),
            "pending": len(self.pending_nolock(es)),
            "words": words,
        }

    @staticmethod
    def pending_nolock(es):
        return [e for e in es if not e.get("processed")]

    def mark_processed(self, timestamps):
        """Flip `processed=True` on the given entries (by ts). This rewrites
        the file, so it must be the *last* write in a batch — called only
        after the batch's DeepSeek pass finished. Timestamps of entries we
        couldn't grade are also marked, so they never block a re-run."""
        ts = set(timestamps)
        es = self.entries()
        changed = False
        for e in es:
            if e.get("ts") in ts and not e.get("processed"):
                e["processed"] = True
                changed = True
        if not changed:
            return
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as fh:
                for e in es:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
