"""The insight pool: the offline learning bank, persisted as JSON on Drive.

Every time a chat session closes, the flow banks ONE candidate golden — the
reasoning trace DeepSeek distilled out of that run, accepted only if the judge
scored it well (good runs only; garbage never enters the pool).

The UI's Offline Learning button then drains the pending pool into an SFT
step. The pool is *append-only until the button is pressed*: collection is
automatic, application is a deliberate human decision, and the button always
works — it shows readiness, it never blocks.

Storage: plain JSON, one file the notebook mounts from Drive, so goldens
survive runtime resets. Thread-safe enough for the Gradio app (a lock around
read-modify-write).
"""

import json
import os
import threading

DEFAULT_THRESHOLD = 30  # goldens collected before the UI suggests applying
MIN_SCORE = 5.0         # judge must score >= this for a run to be banked


class InsightPool:
    def __init__(self, path, threshold=DEFAULT_THRESHOLD,
                 min_score=MIN_SCORE, teacher=None):
        self.path = path
        self.threshold = threshold
        self.min_score = min_score
        self.teacher = teacher          # injected; keeps this module API-clean
        self._lock = threading.Lock()
        self._items = []
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._items = data if isinstance(data, list) else []
        except (OSError, ValueError):
            self._items = []

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._items, fh, ensure_ascii=False, indent=2)

    # -- the one write path ------------------------------------------------

    def bank(self, issue, messages):
        """Judge a closed session, and if it was good, append its golden trace.
        Returns a short human status string for the UI.
        """
        if self.teacher is None:
            return "pool: no teacher bound — skipping"
        score = self.teacher.grade(messages)
        if score < self.min_score:
            return f"pool: session judged {score:.1f} — not banked"
        golden = self.teacher.extract_golden(issue, messages)
        if golden is None:
            return "pool: no golden extracted — not banked"
        with self._lock:
            self._items.append(golden)
            self._save()
        return f"pool: banked 1 golden (judged {score:.1f}), now {len(self._items)}"

    # -- the read path (applies the drain policy) ----------------------------

    def status(self):
        """Count + estimated goldens/tokens for the UI's readiness line."""
        tokens = sum(len(g.get("golden", "").split()) for g in self._items)
        return {"count": len(self._items), "tokens": tokens,
                "threshold": self.threshold,
                "ready": len(self._items) >= self.threshold}

    def drain(self):
        """Atomically take all pending goldens (used by the SFT step)."""
        with self._lock:
            taken = list(self._items)
            self._items = []
            self._save()
        return taken