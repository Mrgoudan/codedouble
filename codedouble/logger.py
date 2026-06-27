"""The logger — the §10 first artifact: faithfully record real interactions.

Capture surfaces (README §6 — record faithfully, reflect later):
  - capture_git()        mine real commits/reverts from a repo (real signal, now)
  - record_interaction() manual API an editor extension / you can call live
  - the post-commit hook (cli install-hook) appends every future commit

`replay()` is the honest evaluation: feed captured moments through a fresh
Double in time order and ask — what WOULD it have done, and would it match what
you actually did? That produces the §8 curve on *your* data.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from typing import List, Optional, Tuple

from .double import Double
from .metrics import Record
from .types import Action, Outcome, Reversibility

DEFAULT_LOG = os.path.join(".codedouble", "interactions.jsonl")

_EXT_LANG = {
    ".py": "python", ".ts": "ts", ".tsx": "ts", ".js": "ts", ".jsx": "ts",
    ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".md": "md", ".cbs": "bsc", ".hbs": "bsc",
    ".rb": "ruby", ".php": "php", ".cs": "csharp",
}


def _lang_of(files: List[str]) -> str:
    c = Counter()
    for f in files:
        _, e = os.path.splitext(f)
        if e in _EXT_LANG:
            c[_EXT_LANG[e]] += 1
    return c.most_common(1)[0][0] if c else ""


def _action_kind(subject: str) -> str:
    s = (subject or "").lower()
    for kw, k in (("revert", "revert"), ("rename", "rename"), ("delete", "delete"),
                  ("remove", "delete"), ("refactor", "refactor"), ("add", "add"),
                  ("fix", "edit")):
        if kw in s:
            return k
    return "edit"


def infer_reversibility(action_kind: str, files: List[str]) -> str:
    hi = action_kind in ("delete", "rename") or any(
        ("migrat" in f.lower() or "schema" in f.lower()) for f in files
    )
    return "high" if hi else "low"


class EventLog:
    """Append-only JSONL store of faithfully-recorded raw moments."""

    def __init__(self, path: str = DEFAULT_LOG):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def read(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def append(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def shas(self) -> set:
        return {r.get("sha") for r in self.read() if r.get("sha")}


def record_interaction(
    log: EventLog,
    request: str,
    resolution: str,
    outcome: str,
    *,
    corrected_from: Optional[str] = None,
    diff: str = "",
    error: str = "",
    lang: str = "",
    files=(),
    action_kind: str = "",
    reversibility: Optional[str] = None,
    source: str = "manual",
) -> dict:
    files = list(files)
    ak = action_kind or _action_kind(request)
    rec = {
        "ts": time.time(), "source": source, "request": request, "diff": diff,
        "error": error, "lang": lang, "files": files, "action_kind": ak,
        "reversibility": reversibility or infer_reversibility(ak, files),
        "outcome": outcome, "resolution": resolution,
        "corrected_from": corrected_from, "sha": None,
    }
    log.append(rec)
    return rec


def capture_git(log: EventLog, repo: str = ".", max_commits: int = 500, quiet: bool = False) -> int:
    """Mine real commits -> resolution events (README §7). Reverts become
    negative labels; other commits are weak 'accepted' signals (committed != endorsed)."""
    try:
        out = subprocess.run(
            ["git", "-C", repo, "log", "-n", str(max_commits),
             "--pretty=format:@@@%H|%ct|%s", "--name-only"],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception as e:  # not a git repo / git missing
        if not quiet:
            print("git log failed:", e)
        return 0
    seen = log.shas()
    added = 0
    for chunk in out.split("@@@"):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        try:
            sha, ct, subject = lines[0].split("|", 2)
        except ValueError:
            continue
        if sha in seen:
            continue
        files = [l for l in lines[1:] if l.strip()]
        ak = _action_kind(subject)
        is_revert = subject.lower().startswith("revert") or ak == "revert"
        rec = {
            "ts": float(ct), "source": "git", "request": subject, "diff": "",
            "error": "", "lang": _lang_of(files), "repo": repo, "files": files[:20],
            "action_kind": ak, "reversibility": infer_reversibility(ak, files),
            "outcome": (Outcome.REVERT.value if is_revert else Outcome.ACCEPTED_SILENT.value),
            "resolution": subject, "corrected_from": None, "sha": sha,
        }
        log.append(rec)
        seen.add(sha)
        added += 1
    if not quiet:
        print(f"captured {added} new git commit(s) -> {log.path}")
    return added


def moment_of(rec: dict) -> dict:
    return {
        "request": rec.get("request", ""), "diff": rec.get("diff", ""),
        "error": rec.get("error", ""), "lang": rec.get("lang", ""),
        "repo": rec.get("repo", ""), "files": tuple(rec.get("files", ())),
        "symbols": tuple(), "action_kind": rec.get("action_kind", ""),
    }


def build_index(raw: List[dict], extractor):
    """Load captured moments into a ResolutionIndex as endorsed precedent
    (used by the live `gate` decision — no replay/scoring needed)."""
    from .index import ResolutionIndex
    from .types import Outcome, ResolutionEvent, Source, next_event_id
    idx = ResolutionIndex()
    for i, rec in enumerate(sorted(raw, key=lambda r: r.get("ts", 0))):
        try:
            oc = Outcome(rec.get("outcome", "accepted_silent"))
        except ValueError:
            oc = Outcome.ACCEPTED_SILENT
        idx.add(ResolutionEvent(
            id=next_event_id(), ts=float(i),
            signature=extractor.extract(moment_of(rec)),
            resolution=rec.get("resolution", ""), outcome=oc,
            source=Source.HUMAN, confidence_at_decision=0.0,
        ))
    return idx


def replay(raw: List[dict], extractor, conf_threshold: float = 0.6) -> Tuple[List[Record], Double]:
    """Counterfactual replay over captured moments in time order."""
    double = Double(extractor, conf_threshold=conf_threshold)
    recs: List[Record] = []
    for i, rec in enumerate(sorted(raw, key=lambda r: r.get("ts", 0))):
        double.now = float(i)
        rev = Reversibility.HIGH if rec.get("reversibility") == "high" else Reversibility.LOW
        dec = double.resolve(moment_of(rec), rev)
        actual = rec.get("resolution", "")
        if dec.action is Action.ASK:
            out, correct = Outcome.ANSWERED, True
        elif dec.resolution == actual:
            out, correct = Outcome.CONFIRMED_GOOD, True
        else:
            out, correct = Outcome.OVERRIDE, False
        recs.append(Record(dec, out, correct))
        double.record(dec, out, corrected_to=(None if out is Outcome.CONFIRMED_GOOD else actual))
    return recs, double
