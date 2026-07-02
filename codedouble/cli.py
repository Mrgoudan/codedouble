"""codedouble CLI — turn it on and watch the effect.

  python3 -m codedouble.cli on            # capture git history + install hook (monitor going forward)
  python3 -m codedouble.cli capture-git   # (re)mine commits into the log
  python3 -m codedouble.cli log ...        # record one real interaction by hand
  python3 -m codedouble.cli report          # replay your real log -> ASCII + report.html
  python3 -m codedouble.cli report --sim    # same visual on the simulated user (guaranteed rich)
  python3 -m codedouble.cli status          # what's captured / is the hook on

Real mode (a real embedder) is the default for `report`.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys

import hashlib
import json
import os
import sys
import time

from .embedder import HashingEmbedder
from .logger import (
    DEFAULT_LOG,
    EventLog,
    _lang_of,
    build_index,
    capture_git,
    codedouble_home,
    moment_of,
    record_interaction,
    replay,
)
from .signature import RuleBasedExtractor
from .viz import ascii_report, render_html


def resolve_ollama_model(model: str = None) -> str:
    """Which local model to use: --model arg > CODEDOUBLE_OLLAMA_MODEL > first
    installed model > 'mistral'. So it 'just works' with whatever you've pulled."""
    if model:
        return model
    env = os.environ.get("CODEDOUBLE_OLLAMA_MODEL")
    if env:
        return env
    from .backends import list_ollama_models
    installed = list_ollama_models()
    return installed[0] if installed else "mistral"


def build_extractor(backend: str, model: str = None):
    backend = (backend or "real").lower()
    if backend == "mistral":
        from .backends import mistral_extractor
        print("[backend] mistral (mistral-embed + LLM extractor)")
        return mistral_extractor()
    if backend in ("ollama", "local"):
        # reasoning on a LOCAL Ollama model + local embedder (no key, no cloud).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from .backends import STEmbedder
            emb = STEmbedder()
            emb_name = "sentence-transformers"
        except Exception:
            emb = HashingEmbedder(256)
            emb_name = "hashing"
        from .backends import ollama_extractor
        m = resolve_ollama_model(model)
        print(f"[backend] ollama (local LLM reasoning: {m}) + {emb_name} retrieval")
        return ollama_extractor(emb, model=m)
    if backend in ("real", "st", "auto"):
        # offline-first: use the cached model if present, NEVER hit the network
        # (avoids the HF HEAD-check + 5x retry hang on an offline machine).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from .backends import STEmbedder
            emb = STEmbedder()
            print("[backend] real: sentence-transformers (CPU, offline/cached)")
            return RuleBasedExtractor(emb)
        except Exception as e:
            print(f"[backend] real embedder not available offline ({type(e).__name__}); "
                  f"using the no-model hashing embedder instead")
            return RuleBasedExtractor(HashingEmbedder(256))
    print("[backend] default: hashing (no model)")
    return RuleBasedExtractor(HashingEmbedder(256))


HOOK = """#!/bin/sh
# codedouble: faithfully record every commit (cheap; analysis happens at report time)
python3 -m codedouble.cli capture-git --quiet >/dev/null 2>&1 || true
"""


def install_hook(repo: str = ".") -> str:
    hook_dir = os.path.join(repo, ".git", "hooks")
    if not os.path.isdir(hook_dir):
        print("not a git repo (no .git/hooks) — skipping hook")
        return ""
    path = os.path.join(hook_dir, "post-commit")
    with open(path, "w") as f:
        f.write(HOOK)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed post-commit hook -> {path}  (monitoring is ON)")
    return path


def cmd_report(args):
    if args.sim:
        from . import demo
        history, index, _ = demo.run(seed=args.seed)
        rules = len(index.semantic.rules)
        sub = "SIMULATED user — validates mechanism, not the thesis. "
    else:
        raw = EventLog(args.log).read()
        if not raw:
            print(f"no interactions captured yet at {args.log}.")
            print("run:  python3 -m codedouble.cli on        (mine git + monitor going forward)")
            print("  or: python3 -m codedouble.cli report --sim   (see it on the simulated user)")
            return
        history, double = replay(raw, build_extractor(args.backend, getattr(args, "model", None)),
                                 conf_threshold=args.conf)
        rules = len(double.index.semantic.rules)
        sub = "REAL captured log. "
    print()
    ascii_report(history, window=args.window, conf_threshold=args.conf)
    out = render_html(history, path=args.out, window=args.window, conf_threshold=args.conf, subtitle=sub)
    print(f"\nrules distilled: {rules}   visual: {os.path.abspath(out)}")


def cmd_capture_git(args):
    n = capture_git(EventLog(args.log), repo=args.repo, quiet=args.quiet)
    if not args.quiet and n:
        print("now run:  python3 -m codedouble.cli report")


def cmd_log(args):
    rec = record_interaction(
        EventLog(args.log), request=args.request, resolution=args.resolution,
        outcome=args.outcome, corrected_from=args.corrected_from, lang=args.lang,
    )
    print("logged:", rec["request"], "->", rec["resolution"], f"({rec['outcome']})")


def cmd_on(args):
    n = capture_git(EventLog(args.log), repo=args.repo)
    install_hook(args.repo)
    print("\ncodedouble is ON. Every commit is now recorded. See the effect with:")
    print("  python3 -m codedouble.cli report")


def _cached_index(log_path, raw, ext):
    """Load a cached index (with vectors) when interactions are unchanged; else
    rebuild + persist. Avoids re-embedding the whole log on every hook call (#5)."""
    from .index import ResolutionIndex
    d = os.path.dirname(log_path) or "."
    cache = os.path.join(d, "index.cache.jsonl")
    meta = cache + ".n"
    try:                                            # mtime key -> appends AND reflect rewrites invalidate
        key = str(os.path.getmtime(log_path))
    except OSError:
        key = str(len(raw))
    try:
        if os.path.exists(cache) and os.path.exists(meta) and open(meta).read().strip() == key:
            return ResolutionIndex.load(cache)
    except Exception:
        pass
    idx = build_index(raw, ext)
    try:
        tmp = cache + ".tmp"
        idx.save(tmp)
        os.replace(tmp, cache)
        with open(meta, "w") as f:
            f.write(key)
    except Exception:
        pass
    return idx


def _decide(log_path, payload, conf):
    """Run one gateway decision: hashing embedder + cached index (Tier 1/2) +
    distilled rules (Tier 3, abstract). The tiered combine lives in Double.resolve."""
    from .double import Double
    from .types import Reversibility
    raw = EventLog(log_path).read()
    ext = RuleBasedExtractor(HashingEmbedder(256))
    idx = _cached_index(log_path, raw, ext)
    idx.semantic.load_rules(os.path.join(os.path.dirname(log_path) or ".", "rules.jsonl"))
    double = Double(ext, idx, conf_threshold=conf)
    double.now = float(len(raw) + 1)
    rev = Reversibility.HIGH if payload.get("reversibility") == "high" else Reversibility.LOW
    return double.resolve(moment_of(payload), rev)


def _log_decision(log_path, event, tool, dec, enforce, intent="", verdict="", before="", after="", target="", session_id="", reason="", cwd=""):
    d = os.path.dirname(log_path) or "."
    path = os.path.join(d, "decisions.jsonl")
    try:
        os.makedirs(d, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({
                "ts": time.time(), "event": event, "tool": tool,
                "session_id": session_id or "",            # scope the panel to ONE session
                "cwd": cwd or "",                          # PAIR the panel to the AI session in this folder
                "target": (target or "").strip()[:160],   # file / command (for reply correlation)
                "intent": (intent or "").strip()[:140],
                "resolution": (dec.resolution or "").strip()[:140],
                "before": (before or "").strip()[:400],   # what the AI proposed
                "after": (after or "").strip()[:600],      # the correction the double asked for
                "reason": (reason or "").strip()[:1200],   # the WHY — the product's voice; generous cap
                "verdict": verdict,            # inject | allow | ask | deny | answered | shadow | watch
                "action": dec.action.value, "ask": dec.action.value == "ask",
                "confidence": round(dec.confidence, 3), "coverage": round(dec.coverage, 3),
                "n": len(dec.retrieved), "enforce": enforce,
            }) + "\n")
        # cap growth (#6): cheap stat each call, rewrite only when over the cap
        if os.path.getsize(path) > 1_500_000:
            with open(path) as f:
                tail = f.readlines()[-1500:]
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.writelines(tail)
            os.replace(tmp, path)
    except Exception:
        pass


_QC_SYSTEM = (
    "You are a guardrail watching an AI coding session. Given the session's ANCHORS "
    "(goal / constraints / decisions / todos / avoid) and ONE proposed change, decide "
    "whether the change CONTRADICTS a specific anchor. Judge CONSISTENCY, not "
    "sufficiency: an atomic, partial, or mechanical step that simply doesn't advance "
    "the goal is FINE — most steps of a large task look insufficient alone. Flag ONLY "
    "a clear violation of one nameable anchor. Return ONLY JSON: "
    "{\"violates\": \"<the exact anchor text violated>\", \"refine\": \"<one concise "
    "instruction for redoing it>\"} if it clearly contradicts one, else "
    "{\"violates\": null}."
)


def _last_prompt(log_path):
    """The most recent user prompt the gateway saw (the real intent for QC)."""
    p = os.path.join(os.path.dirname(log_path) or ".", "decisions.jsonl")
    last = ""
    try:
        with open(p) as f:
            lines = f.readlines()[-800:]          # tail only — don't scan the whole log
        for line in lines:
            try:
                r = json.loads(line)
                if r.get("event") == "UserPromptSubmit" and r.get("intent"):
                    last = r["intent"]
            except Exception:
                pass
    except Exception:
        pass
    return last


_COMMENT_LINE_RE = re.compile(r"^\s*(#|//|/\*|\*/|\*|<!--|--\s)")


def _comment_only(text):
    """A change consisting solely of comments cannot violate a behavioral anchor —
    allow it without any judge (kills the mention-vs-use false-positive class at
    the source, deterministically)."""
    lines = [l for l in str(text).splitlines() if l.strip()]
    return bool(lines) and all(_COMMENT_LINE_RE.match(l) for l in lines)


def _suspicious(anchors, proposed):
    """Cheap lexical trigger for a remote second opinion: the change shares >=2
    distinctive tokens with a single constraint/decision/avoid item. Only a trigger
    (costs one remote call), never a verdict."""
    psig = _anchor_sig(str(proposed)[:2000])
    for key in ("constraints", "decisions", "avoid"):
        for item in (anchors.get(key) or []):
            if len(psig & _anchor_sig(item)) >= 2:
                return True
    return False


def _quality_check(anchors, proposed):
    """Does this change CONTRADICT a specific session anchor? Returns
    (ok, violated_anchor, refine). Deny requires a NAMEABLE violation — consistency,
    not sufficiency. CASCADED judging (calibrated 2026-07: local-7B FP 12% / FN 19%,
    remote judge got every miss right): comment-only -> allow deterministically;
    the LOCAL model is the rough filter; the REMOTE judge confirms only when the
    verdict is about to matter — a local DENY (before we gate the AI) or a local
    allow on a lexically anchor-adjacent change. Plain allows never pay a remote
    call. Fail-open at every step (never block on judge failure)."""
    if _comment_only(proposed):
        return True, "", ""

    def _judge(prefer):
        from .backends import llm_complete
        out = llm_complete(json.dumps({"anchors": anchors, "proposed_change": proposed[:2000]}),
                           system=_QC_SYSTEM, json_mode=True, timeout=45, prefer=prefer)
        data = json.loads(out)
        violated = str(data.get("violates") or "").strip()
        if not violated or violated.lower() in ("null", "none"):
            return True, "", ""
        return False, violated[:300], str(data.get("refine", ""))[:600]

    try:
        ok, violated, refine = _judge("local")
    except Exception:
        return True, "", ""
    try:
        if not ok:
            return _judge("remote")            # confirm every send-back; overturn local FPs
        if _suspicious(anchors, proposed):
            return _judge("remote")            # second opinion on anchor-adjacent allows
    except Exception:
        pass                                   # remote unreachable -> keep the local verdict
    return ok, violated, refine


def _maybe_reflect(log_path, summarize=False):
    """Trigger reflection when something that MATTERS happens (a correction, a
    send-back, an override) — rate-limited (coalesce bursts) and DETACHED so the hook
    returns immediately. Event-driven, not idle/clock-driven."""
    d = os.path.dirname(log_path) or "."
    marker = os.path.join(d, ".last_reflect")
    now = time.time()
    try:
        last = float(open(marker).read().strip())
    except Exception:
        last = 0.0
    if now - last < 20:                      # coalesce bursts
        return
    try:
        with open(marker, "w") as f:
            f.write(str(now))
    except Exception:
        pass
    import subprocess
    base = [sys.executable, "-m", "codedouble.cli"]
    cmds = [base + ["reflect", "--quiet"]]
    if summarize:
        cmds.append(base + ["summarize", "--quiet"])
    for argv in cmds:
        try:
            subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        except Exception:
            pass


# the conversational signal we were missing: you correcting the AI's UNDERSTANDING
_CORRECTION_RE = re.compile(
    r"(^\s*(no\b|nope\b|actually\b|wrong\b))|"
    r"(i meant|i mean\b|you mis|misunderstood|understood me|that'?s not|not what i|"
    r"you'?re wrong|you got it wrong|instead of|rather than|the point is|i said\b)", re.I)


def _last_assistant_text(transcript_path):
    """Best-effort: the AI's previous turn (what the correction is aimed at)."""
    if not transcript_path:
        return ""
    try:
        with open(transcript_path) as f:
            lines = f.readlines()[-40:]
        for line in reversed(lines):
            try:
                d = json.loads(line)
            except Exception:
                continue
            msg = d.get("message") or d
            role = d.get("type") or d.get("role") or msg.get("role")
            if role == "assistant":
                c = msg.get("content")
                if isinstance(c, str):
                    return c[:300]
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return (part.get("text") or "")[:300]
    except Exception:
        pass
    return ""


def _capture_correction(log_path, prompt, transcript_path):
    """When your message CORRECTS the AI's understanding (not a new task), record it
    as a gold intent-correction — the conceptual misunderstanding signal the
    code-action capture misses. Heuristic for now; an LLM classifier would be precise."""
    p = (prompt or "").strip()
    if len(p) < 4 or not _CORRECTION_RE.search(p[:160]):
        return
    ai_last = _last_assistant_text(transcript_path)
    try:
        record_interaction(
            EventLog(log_path), request=p[:140], resolution=p[:200], outcome="override",
            corrected_from=(ai_last[:200] if ai_last else None),
            lang="", files=[], action_kind="clarify", source="chat",
        )
        _maybe_reflect(log_path, summarize=True)   # a correction matters -> reflect now
    except Exception:
        pass


def _capture_reply(log_path, tool, target, ran):
    """The double sent the AI back on this action and it has now completed — record the
    OUTCOME as a high-quality signal (you only ever talk to the AI): ANSWERED if the
    action stuck unchanged (the double over-intervened -> weaken that rule), OVERRIDE if
    it was redone differently (the redo is the true intent)."""
    if not target:
        return
    dpath = os.path.join(os.path.dirname(log_path) or ".", "decisions.jsonl")
    now = time.time()
    try:
        with open(dpath) as f:
            lines = f.readlines()[-600:]
    except Exception:
        return
    inter, handled = None, False
    for line in lines:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("tool") != tool or r.get("target") != target:
            continue
        v = r.get("verdict")
        if v in ("ask", "deny") and (now - r.get("ts", 0)) < 900:
            inter, handled = r, False        # a fresh intervention to attribute
        elif v == "answered":
            handled = True                   # already attributed
    if inter is None or handled:
        return
    proposed = (inter.get("before") or "").strip()
    rann = (ran or "").strip()
    def norm(s):
        return " ".join(s.split())[:200]
    if proposed and norm(rann) != norm(proposed):
        outcome, resolution, corrected = "override", rann, proposed   # redone differently
    else:
        outcome, resolution, corrected = "answered", (rann or inter.get("intent", "")), None
    files = [target] if tool in ("Edit", "Write") else []
    try:
        record_interaction(
            EventLog(log_path), request=inter.get("intent", f"{tool} {target}"),
            resolution=(resolution or "")[:200], outcome=outcome,
            corrected_from=(corrected[:200] if corrected else None),
            lang=_lang_of(files), files=files, action_kind="edit", source="reply",
        )
    except Exception:
        pass
    try:                                     # mark handled so we attribute once
        with open(dpath, "a") as f:
            f.write(json.dumps({
                "ts": now, "event": "PreToolUse", "tool": tool, "target": (target or "")[:160],
                "intent": inter.get("intent", ""), "resolution": (resolution or "")[:140],
                "before": "", "after": "", "verdict": "answered",
                "action": "act", "ask": False, "confidence": 0, "coverage": 0, "n": 0, "enforce": True,
            }) + "\n")
    except Exception:
        pass
    _maybe_reflect(log_path)               # a reply/override matters -> reflect now


# ---- incremental per-session summary; the gate checks AI actions against it ----
def _session_dir(log_path):
    d = os.path.join(os.path.dirname(log_path) or ".", "sessions")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _sid_file(log_path, sid, ext):
    return os.path.join(_session_dir(log_path), f"{(sid or 'default')[:64]}{ext}")


def _session_outcome(log_path, sid, tool, target):
    """Append a completed action to this session's OUTCOMES stream — the evidence the
    anchor maintainer uses to clear done todos / confirm decisions (the maintainer
    otherwise sees only prompts, so todos would never self-clear). Compact: tool +
    target only, consecutive duplicates skipped."""
    if not sid or not (target or "").strip():
        return
    try:
        p = _sid_file(log_path, sid, ".outcomes.jsonl")
        rec = {"ts": time.time(), "tool": tool, "target": str(target).strip()[:120]}
        try:
            with open(p) as f:
                last = json.loads(f.readlines()[-1])
            if last.get("tool") == rec["tool"] and last.get("target") == rec["target"]:
                return                       # consecutive duplicate (retry/loop) -> skip
        except Exception:
            pass
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _flag_bypassed_sendback(log_path, sid, tool, target, ran, window_s=1800):
    """FALSE-POSITIVE labeller for send-back precision: when a change the double DENIED
    subsequently lands anyway via another path (e.g. an Edit deny re-applied via Bash),
    the send-back demonstrably didn't change the outcome — count it as fought/bypassed.
    Deterministic (target match or strong content overlap), log-only, self-labelling:
    no user labelling needed. Appends a verdict="bypassed" meta-record referencing the
    deny's ts; metrics/panel can then compute precision = 1 - bypassed/denied."""
    if not sid or not (ran or "").strip():
        return
    try:
        p = os.path.join(os.path.dirname(log_path) or ".", "decisions.jsonl")
        rows = [json.loads(l) for l in open(p).readlines()[-200:] if l.strip()]
    except Exception:
        return
    now = time.time()
    flagged = {r.get("ref") for r in rows if r.get("verdict") == "bypassed"}
    rsig = _anchor_sig(str(ran)[:2000])
    for r in reversed(rows):
        if r.get("verdict") != "deny" or r.get("session_id") != sid:
            continue
        if r.get("ts", 0) < now - window_s or r.get("ts") in flagged:
            continue
        dsig = _anchor_sig(r.get("before", ""))
        same_target = bool(r.get("target")) and (r["target"] in str(target) or str(target) in r["target"])
        overlap = bool(dsig) and len(rsig & dsig) >= max(4, int(0.6 * len(dsig)))
        if (same_target and overlap) or (overlap and len(dsig) >= 8):
            try:
                with open(p, "a") as f:
                    f.write(json.dumps({
                        "ts": now, "event": "PostToolUse", "tool": tool, "verdict": "bypassed",
                        "session_id": sid, "target": str(target)[:160], "ref": r.get("ts"),
                        "reason": ("[codedouble] a change denied earlier landed anyway via "
                                   f"{tool} — that send-back is counted as a false positive"),
                    }) + "\n")
            except Exception:
                pass
            return                       # flag at most one deny per completed action


def _session_note(log_path, sid, prompt, cwd=None):
    """Incrementally append the user's intent (and the cwd / project it came from)
    to this session's running notes — the cwd is what scopes graduation/seeding."""
    try:
        rec = {"ts": time.time(), "prompt": (prompt or "")[:500]}
        if cwd:
            rec["cwd"] = cwd
        with open(_sid_file(log_path, sid, ".jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


# injected IDE/system context ("<ide_selection>...", "<ide_opened_file>...",
# "<system-reminder>...") is NOT the developer's intent — strip it so it can never
# become a goal/anchor ("goal will never be a selection"). Closed blocks are removed
# in place; an unclosed/truncated block is stripped to end.
_INJECTED_RE = re.compile(
    r"<(ide_selection|ide_opened_file|system-reminder|system)\b[^>]*>.*?(</\1>|$)", re.I | re.S)


def _clean_prompt(text):
    # drop a re-pasted anchors block (the injected "[codedouble] Session anchors ..."
    # context) — keep only the developer's own words before it. This is what leaked
    # another task's anchors (F40) into an unrelated session.
    t = re.split(r"\[codedouble\] Session anchors", str(text or ""))[0]
    return re.sub(r"\s+", " ", _INJECTED_RE.sub(" ", t)).strip()


def _heuristic_goal(notes):
    """A lightweight goal from the opening prompt (it usually frames the task) — so
    the session goal still shows when the LLM is unavailable (rate-limited/offline)."""
    for n in notes:
        t = _clean_prompt(n)                           # drop injected IDE/system noise
        if t[:1] == "[":
            continue                                   # [outcomes]/bracketed evidence, not intent
        if len(t) >= 12:
            t = re.split(r"(?<=[.!?])\s", t)[0]       # first sentence
            t = re.split(r"\s+at\s+[~/]", t)[0]        # drop "... at /a/path"
            t = re.sub(r"\s*\([^)]*\)", "", t)         # drop parentheticals
            return t.strip(" .,:")[:140]
    return ""


_ANCHOR_KEYS = ("goal", "constraints", "decisions", "todos", "avoid")

_UPDATE_SYSTEM = (
    "You maintain the ANCHORS of a coding session as it evolves. You are given the "
    "CURRENT anchors (JSON) and the developer's NEW message(s). Return the UPDATED anchors "
    "as JSON with EXACTLY these keys: {\"goal\": \"<one sentence>\", \"constraints\": [], "
    "\"decisions\": [], \"todos\": [], \"avoid\": []}. Apply the SMALLEST edit that "
    "reflects the new messages: ADD a constraint/decision/todo/avoid only when a message "
    "introduces one; UPDATE (rewrite in place) an item a message refines or corrects; "
    "DELETE an item a message supersedes, contradicts, completes, or resolves; keep the "
    "goal unchanged unless a message clearly changes the overall goal. If the new messages "
    "change nothing (acknowledgement, question, chit-chat), return the CURRENT anchors "
    "unchanged. A message starting with [outcomes] lists actions that were actually "
    "completed — it is EVIDENCE, never new intent: DELETE a todo only when an outcome "
    "explicitly shows that exact task finished; KEEP every todo the outcomes do not "
    "cover, and never remove constraints/decisions because of outcomes alone. Each "
    "item one short line, grounded in the messages; never invent content."
)


def _norm_anchors(a, fallback=None):
    """Coerce an anchors dict to the canonical shape; fall back per-key on bad values."""
    fb = fallback or {}
    out = {"goal": str(a.get("goal") or fb.get("goal") or "")}
    for k in ("constraints", "decisions", "todos", "avoid"):
        v = a.get(k)
        out[k] = ([str(x).strip() for x in v if str(x).strip()][:12]
                  if isinstance(v, list) else list(fb.get(k) or []))
    return out


def _update_anchors(current, new_messages):
    """Mem0-style INCREMENTAL anchor maintenance: given the current anchors and ONLY the
    NEW messages, edit in place (add / update / delete / noop) — never re-derive from full
    history. Local model first (cheap, per-turn). Returns (anchors, ok): ok=False when the
    LLM was unreachable, so the caller keeps those notes unprocessed and retries later
    (current anchors are preserved — a fallback pass never wipes them)."""
    msgs = [c for m in (new_messages or []) if (c := _clean_prompt(m))]   # strip IDE/system noise
    cur = {k: current.get(k) for k in _ANCHOR_KEYS}
    if not msgs:
        return _norm_anchors(cur), True
    try:
        from .backends import llm_complete
        out = llm_complete(json.dumps({"current": cur, "new_messages": msgs}),
                           system=_UPDATE_SYSTEM, json_mode=True, timeout=60, prefer="local")
        a = json.loads(out)
        if isinstance(a, dict):
            return _norm_anchors(a, fallback=cur), True
    except Exception:
        pass
    kept = _norm_anchors(cur, fallback=cur)
    if not kept["goal"]:
        kept["goal"] = _heuristic_goal(msgs)                     # show a goal even offline
    return kept, False


def _render_anchors(a):
    parts = []
    if a.get("goal"):
        parts.append("Goal: " + str(a["goal"]))
    for label, key in (("Constraints", "constraints"), ("Decisions made", "decisions"),
                       ("TODO", "todos"), ("Avoid", "avoid")):
        items = a.get(key) or []
        if items:
            parts.append(label + ":\n" + "\n".join("  - " + str(x)[:160] for x in items[:8]))
    return "\n".join(parts)[:1600]


def _repo_root(start):
    """Nearest ancestor of `start` containing .git, else `start` itself. The stable
    per-PROJECT identity. Pure-Python walk-up — no subprocess on the hook hot path."""
    try:
        cur = os.path.abspath(start or ".")
    except Exception:
        return ""
    here = cur
    while True:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return here
        cur = parent


def _scope_key(cwd):
    """Stable per-PROJECT bucket key (repo root of cwd). Separating anchors by
    PROJECT — not by ephemeral session id — keeps cross-session learning WITHIN a
    project while making cross-domain contamination structurally impossible: another
    project's anchors live in another bucket and can never be seeded here."""
    root = _repo_root(cwd) if cwd else ""
    if not root:
        return "default"
    # readable base + a hash of the FULL path so distinct paths never collide to one
    # bucket (e.g. /a/b vs /a-b) and the name stays filesystem-bounded.
    base = re.sub(r"[^A-Za-z0-9_.-]", "-", root.strip("/"))
    h = hashlib.sha1(root.encode("utf-8")).hexdigest()[:8]
    return (base[:80] + "-" + h) if base else "default"


def _scope_anchors_path(log_path, scope):
    """Per-project durable-anchor file: <store>/anchors/<scope>.json."""
    d = os.path.join(os.path.dirname(log_path) or ".", "anchors")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, f"{scope or 'default'}.json")


def _global_distill_path(log_path):
    """The single GLOBAL distill — durable rules that recur across projects."""
    return os.path.join(os.path.dirname(log_path) or ".", "anchors", "__global__.json")


def _promote_global(log_path, min_projects=2):
    """Promote durable rules that RECUR across >= min_projects project distills into
    the one global distill. A project-specific rule (e.g. one project's include
    paths) appears in a single bucket and stays local; only genuinely cross-cutting
    habits globalize. Goal/decisions never globalize (session/project scoped)."""
    import glob as _glob
    d = os.path.join(os.path.dirname(log_path) or ".", "anchors")
    out = {}
    for key in ("constraints", "avoid"):
        texts, projects = {}, {}            # token-sig -> longest text / set(project files)
        for f in _glob.glob(os.path.join(d, "*.json")):
            if os.path.basename(f) == "__global__.json":
                continue                    # skip the global distill itself
            try:
                g = json.loads(open(f).read())
            except Exception:
                continue
            items = g.get(key)
            if not isinstance(items, list):
                continue                    # tolerate a malformed/partial distill file
            for item in items:
                if not isinstance(item, str):
                    continue
                sig = _anchor_sig(item)
                if not sig:
                    continue
                projects.setdefault(sig, set()).add(f)
                if len(item) > len(texts.get(sig, "")):
                    texts[sig] = item
        promoted = _dedupe_anchors([texts[s] for s in texts
                                    if len(projects[s]) >= min_projects])
        if promoted:
            out[key] = promoted[-24:]
    gpath = _global_distill_path(log_path)
    try:
        tmp = gpath + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(out))
        os.replace(tmp, gpath)
    except Exception:
        pass


_ANCHOR_STOP = {
    "the", "and", "for", "with", "use", "all", "any", "only", "not", "are", "was",
    "you", "your", "its", "this", "that", "these", "those", "must", "should",
}


def _stem(w):
    """Crude morphological fold so run/runs, simulate/simulation/simulations,
    guess/guessing collapse to one token — enough for near-dup detection."""
    for suf in ("ations", "ation", "ings", "ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _anchor_tokens(s):
    """Distinctive, lightly-stemmed tokens of an anchor (drops tiny/stopwords)."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower())
            if len(w) >= 3 and w not in _ANCHOR_STOP]


def _anchor_sig(s):
    return frozenset(_anchor_tokens(s))


def _dedupe_anchors(items, thresh=0.7):
    """Collapse exact / substring / high-overlap (Jaccard>=thresh) near-duplicates,
    preferring the more complete phrasing. Deterministic — catches the
    path/flag/paraphrase pile-up that exact-match dedup let through, and drops
    truncated fragments (a substring of a kept item). Semantic paraphrase families
    that share little vocabulary are left for the LLM pass."""
    kept = []  # (text, sig)
    for raw in items:
        s = re.sub(r"\s+", " ", str(raw)).strip()
        if not s:
            continue
        sig = _anchor_sig(s)
        sl = s.lower()
        hit = None
        for i, (t, tsig) in enumerate(kept):
            tl = t.lower()
            if sl == tl or sl in tl or tl in sl:
                hit = i; break
            if sig and tsig:
                inter = len(sig & tsig)
                if not inter:
                    continue
                jaccard = inter / len(sig | tsig)
                # containment catches paraphrases where one anchor's content is
                # ~subsumed by the other ("do real runs, no simulation" vs "real
                # runs only — no simulation or guessing") — Jaccard alone misses these.
                contain = inter / min(len(sig), len(tsig))
                if jaccard >= thresh or (inter >= 3 and contain >= 0.8):
                    hit = i; break
        if hit is None:
            kept.append((s, sig))
        elif len(s) > len(kept[hit][0]):       # keep the more complete phrasing
            kept[hit] = (s, sig)
    return [t for t, _ in kept]


_DEDUPE_SYSTEM = (
    "You are consolidating a list of durable {kind} accumulated across many coding "
    "sessions. Collapse near-duplicates and paraphrases into ONE crisp line each, "
    "drop truncated or garbled fragments, and keep only genuinely distinct items. "
    "Preserve the EXACT text of any file paths, flags, or commands. "
    'Return ONLY JSON: {{"items": ["...", "..."]}}.'
)


def _llm_dedupe_anchors(items, kind):
    """Best-effort semantic consolidation (collapses paraphrase families like
    'real runs only' vs 'do real runs, no simulation' that lexical dedup can't).
    HARD, session-wide -> remote tier; falls back to the heuristic result."""
    if len(items) < 4:
        return items
    try:
        from .backends import llm_complete
        out = llm_complete(json.dumps(items), system=_DEDUPE_SYSTEM.format(kind=kind),
                           json_mode=True, timeout=60, prefer="remote")
        arr = json.loads(out)
        if isinstance(arr, dict):
            arr = arr.get("items") or next((v for v in arr.values() if isinstance(v, list)), [])
        cleaned = [str(x).strip() for x in arr if str(x).strip()]
        return cleaned if cleaned else items
    except Exception:
        return items


def _merge_project_distill(log_path, a, scope="default"):
    """Graduate a session's durable CONSTRAINTS/DECISIONS/AVOID into its PROJECT
    distill (goal-free: a goal is session-only, since drift is per-conversation).
    Heuristic + best-effort LLM dedupe so paraphrases don't pile up; the global
    tier is built separately by _promote_global."""
    if not isinstance(a, dict):
        return
    gp = _scope_anchors_path(log_path, scope)
    try:
        g = json.loads(open(gp).read()) if os.path.exists(gp) else {}
    except Exception:
        g = {}
    for key in ("constraints", "decisions", "avoid"):
        combined = list(g.get(key) or []) + list(a.get(key) or [])
        merged = _llm_dedupe_anchors(_dedupe_anchors(combined), key)
        g[key] = merged[-24:]
    # goal is SESSION-ONLY: it never enters a distill tier (drift is per-conversation)
    try:
        tmp = gp + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(g))
        os.replace(tmp, gp)         # atomic — the live hooks graduate concurrently
    except Exception:
        pass


def _session_scope(log_path, sid):
    """The PROJECT scope this session belongs to — derived from the cwd recorded in
    its notes (most recent wins). Lets graduation and seeding stay per-project."""
    cwd = ""
    try:
        for l in open(_sid_file(log_path, sid, ".jsonl")).readlines()[-20:]:
            c = json.loads(l).get("cwd")
            if c:
                cwd = c
    except Exception:
        pass
    return _scope_key(cwd)


def _decision_anchors(log_path, sid):
    """The ONE anchor view every decision consumes — injection, drift, and QC alike:
    this session's OWN anchors (the goal comes ONLY from here) combined with the
    project distill and the global distill, deduped, session items first. This is
    what makes cross-session learning BITE at the moment of action, automatically —
    the user never manually curates memory to get good behavior. A fresh session is
    thereby primed from its project's rules; a distill never contributes a goal
    (a prior session's goal is not this session's task)."""
    own = _session_anchors_struct(log_path, sid)

    def _load(p):
        try:
            return json.loads(open(p).read()) if os.path.exists(p) else {}
        except Exception:
            return {}

    scope = _session_scope(log_path, sid)
    proj = _load(_scope_anchors_path(log_path, scope))
    glob = _load(_global_distill_path(log_path))
    out = {"goal": str(own.get("goal") or ""), "todos": list(own.get("todos") or [])}
    for key, tiers in (("constraints", (own, proj, glob)),
                       ("decisions", (own, proj)),
                       ("avoid", (own, proj, glob))):
        items = []
        for t in tiers:
            items.extend(t.get(key) or [])
        out[key] = _dedupe_anchors(items)[:12]
    return out


def _session_anchors(log_path, sid):
    """Anchors (rendered) for steering — the merged decision view (session ⊕ project
    distill ⊕ global distill). A session with no own anchors gets its project's
    rules and no goal; one with anchors gets both, deduped."""
    return _render_anchors(_decision_anchors(log_path, sid))


def _session_anchors_struct(log_path, sid):
    """The session's OWN structured anchors (dict) — the drift reference. Only the
    session's own (seeded distills carry no goal); {} if not consolidated yet."""
    try:
        ap = _sid_file(log_path, sid, ".anchors.json")
        if os.path.exists(ap):
            return json.loads(open(ap).read())
    except Exception:
        pass
    return {}


_DRIFT_SYSTEM = (
    "You are a guardrail watching ONE coding session for DRIFT. Given the session's "
    "GOAL, hard CONSTRAINTS, DECISIONS already made, and approaches to AVOID, then the "
    "action the AI is about to take, decide if the action DRIFTS: it works toward "
    "something other than the goal, breaks a constraint, contradicts a decision, or does "
    "something on the avoid list. Be conservative \u2014 setup/exploration that plausibly "
    "serves the goal is NOT drift, and MENTIONING, documenting, or discussing an avoided "
    "approach is NOT drift; only actually DOING it is. Return ONLY JSON: "
    '{"drift": true|false, "reason": "<one line>", "redirect": "<what to do instead>"}.'
)


def _drift_check(anchors, request, diff=""):
    """Is the AI's action drifting from THIS session's established anchors? Cheap
    deterministic avoid-list hit first (free), then a best-effort LLM judge (frequent
    per-action -> local model first). Returns (is_drift, reason, redirect).
    Conservative: with no established goal it never flags, so the double never blocks
    legitimate exploration (act on evidence, not ignorance)."""
    if not isinstance(anchors, dict) or not anchors.get("goal"):
        return (False, "", "")
    action = (str(request) + "\n" + str(diff)).strip()
    asig = _anchor_sig(action)
    act = re.sub(r"\s+", " ", action.lower())
    for av in (anchors.get("avoid") or []):
        sig = _anchor_sig(av)
        avn = re.sub(r"\s+", " ", str(av).lower()).strip()
        # high-precision: the avoid PHRASE literally appears (catches short avoids the
        # token test alone misses, e.g. an echo of "use global state"), OR a >=3-token
        # paraphrase overlap. Avoids BOTH the false positives and the short-avoid dead zone.
        if (len(avn) >= 8 and avn in act) or len(asig & sig) >= 3:
            return (True, "on the avoid list: " + str(av), "avoid this: " + str(av))
    try:
        from .backends import llm_complete
        prompt = ("GOAL: %s\nCONSTRAINTS: %s\nDECISIONS: %s\nAVOID: %s\n\nACTION:\n%s" % (
            anchors.get("goal", ""), anchors.get("constraints") or [],
            anchors.get("decisions") or [], anchors.get("avoid") or [], action[:1200]))
        d = json.loads(llm_complete(prompt, system=_DRIFT_SYSTEM, json_mode=True,
                                    timeout=30, prefer="local"))
        if isinstance(d, dict) and d.get("drift"):
            try:                                   # local judge is noisy -> remote confirms
                d2 = json.loads(llm_complete(prompt, system=_DRIFT_SYSTEM, json_mode=True,
                                             timeout=45, prefer="remote"))
                if isinstance(d2, dict) and not d2.get("drift"):
                    return (False, "", "")         # overturned: don't send the AI back
                d = d2 if isinstance(d2, dict) and d2.get("drift") else d
            except Exception:
                pass
            return (True, str(d.get("reason", ""))[:400], str(d.get("redirect", ""))[:400])
    except Exception:
        pass
    return (False, "", "")


def _session_summary(log_path, sid):
    """What the gate checks AI actions against: the structured anchors if
    consolidated, else the recent running notes."""
    a = _session_anchors(log_path, sid)
    if a:
        return a
    try:
        lines = open(_sid_file(log_path, sid, ".jsonl")).readlines()[-12:]
        return "\n".join("- " + json.loads(l).get("prompt", "")[:160] for l in lines)[:1600]
    except Exception:
        return ""


def cmd_summarize(args):
    """Maintain each active session's ANCHORS incrementally — Mem0-style add/update/
    delete/noop over ONLY the new notes since last update (tracked by `_n`), not a
    re-derivation from full history. Local model; preserves anchors on LLM failure and
    retries the unprocessed notes later. Idempotent; run coalesced/detached."""
    import glob
    n = 0
    for notes_path in glob.glob(os.path.join(_session_dir(args.log), "*.jsonl")):
        try:
            notes = [json.loads(l).get("prompt", "") for l in open(notes_path) if l.strip()]
        except Exception:
            continue
        if not notes:
            continue
        apath = notes_path[:-6] + ".anchors.json"
        try:
            cur = json.loads(open(apath).read())
        except Exception:
            cur = {}
        processed = int(cur.get("_n") or 0)
        new = list(notes[processed:])
        # outcome loop: what actually got DONE since the last update (clears todos)
        try:
            outs = [json.loads(l) for l in open(notes_path[:-6] + ".outcomes.jsonl") if l.strip()]
        except Exception:
            outs = []
        processed_o = int(cur.get("_o") or 0)
        new_out = outs[processed_o:]
        if not new and not new_out:
            continue                      # nothing new -> skip (exact counts, no mtime race)
        if new_out:
            new.append("[outcomes] actions completed since the last update: " + "; ".join(
                f"{o.get('tool', '?')} {o.get('target', '')}" for o in new_out[-30:]))
        updated, ok = _update_anchors(cur, new)
        updated["_n"] = len(notes) if ok else processed   # advance only when the LLM ran
        updated["_o"] = len(outs) if ok else processed_o
        sid = os.path.basename(notes_path)[:-6]
        try:
            with open(apath, "w") as f:
                f.write(json.dumps(updated))
            _merge_project_distill(args.log, updated, _session_scope(args.log, sid))
            n += 1
        except Exception:
            pass
    _promote_global(args.log)              # recurring rules -> the global distill
    if not args.quiet:
        print(f"updated {n} session(s)")


def cmd_gate(args):
    """The gateway decision: read an intent (JSON on stdin), emit a decision JSON.
    Used for BOTH directions: intake -> use `inject`; outtake -> use `allow`/`ask`."""
    from .types import Action
    payload = {}
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read().strip()
            if data:
                payload = json.loads(data)
    except Exception:
        payload = {}
    if args.request:
        payload.setdefault("request", args.request)
    dec = _decide(args.log, payload, args.conf)
    inject = ""
    if dec.retrieved:
        inject = (f"[codedouble] precedent for this kind of change: '{dec.resolution}' "
                  f"(confidence {dec.confidence:.2f}, {len(dec.retrieved)} similar past decisions).")
    print(json.dumps({
        "action": dec.action.value,
        "allow": (True if args.shadow else dec.action is not Action.ASK),
        "ask": dec.action is Action.ASK and not args.shadow,
        "resolution": dec.resolution, "confidence": round(dec.confidence, 3),
        "coverage": round(dec.coverage, 3), "shadow": bool(args.shadow),
        "inject": inject, "rationale": dec.rationale,
    }))


def cmd_hook(args):
    """Claude Code hook adapter: read the hook event JSON on stdin, run the gate,
    emit Claude Code's expected output. FAIL-OPEN (never breaks your session) and
    SHADOW by default (no blocking) unless CODEDOUBLE_ENFORCE=1.

    intake (UserPromptSubmit) -> inject precedent as additionalContext (always)
    outtake (PreToolUse)      -> shadow: log only; enforce: allow / ask via 2x2
    """
    from .types import Action
    try:
        data = sys.stdin.read()
        ev = json.loads(data) if data.strip() else {}
    except Exception:
        return  # fail-open
    try:
        name = ev.get("hook_event_name", "")
        enforce = os.environ.get("CODEDOUBLE_ENFORCE") == "1"
        cwd = ev.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or ""    # project scope (both paths)

        if name == "UserPromptSubmit":
            prompt = ev.get("prompt", "")
            _capture_correction(args.log, prompt, ev.get("transcript_path"))   # learn from "no, I meant X"
            _session_note(args.log, ev.get("session_id", ""), prompt, cwd)     # incremental session summary
            dec = _decide(args.log, {"request": prompt, "reversibility": "low"}, args.conf)
            injected = bool(dec.retrieved and dec.confidence >= 0.4 and dec.resolution)
            _log_decision(args.log, name, None, dec, enforce, intent=prompt,
                          verdict=("inject" if injected else "watch"),
                          session_id=ev.get("session_id", ""), cwd=cwd)
            # maintain the session anchors INCREMENTALLY (Mem0-style add/update/delete/
            # noop over the new notes) — coalesced >=20s, detached, on the local model.
            # No importance gate: a trivial turn is a NOOP; a real change edits in place.
            _maybe_reflect(args.log, summarize=True)
            blocks = []
            anchors = _session_anchors(args.log, ev.get("session_id", ""))
            if anchors:
                blocks.append("[codedouble] Session anchors — stay within these (goal / constraints / "
                              "decisions already made / todos):\n" + anchors)
            if injected:
                blocks.append(f"[codedouble] The developer has consistently preferred '{dec.resolution}' "
                              f"for this kind of request ({len(dec.retrieved)} precedents, "
                              f"confidence {dec.confidence:.2f}). Apply it unless this case clearly differs.")
            if blocks:
                print(json.dumps({"hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit", "additionalContext": "\n\n".join(blocks)}}))
            return

        if name == "PreToolUse":
            tool = ev.get("tool_name", "")
            ti = ev.get("tool_input", {}) or {}
            blob = (str(ti.get("command", "")) + " " + str(ti.get("file_path", ""))).lower()
            rev = "high" if any(k in blob for k in (
                "rm ", "rm -", "drop ", "delete", "truncate", "migrat", "schema",
                "force", "reset --hard")) else "low"
            payload = {
                "request": f"{tool} {ti.get('file_path','')} {ti.get('command','')}".strip(),
                "diff": str(ti.get("new_string", "") or ti.get("command", "")),
                "files": [ti["file_path"]] if ti.get("file_path") else [],
                "reversibility": rev,
            }
            dec = _decide(args.log, payload, args.conf)
            before = str(ti.get("new_string") or ti.get("content") or ti.get("command") or "")
            target = str(ti.get("file_path") or ti.get("command") or "")
            verdict, reason, after = "shadow", f"[codedouble] {dec.rationale}", ""
            # KNOWN-BAD: you've OVERRIDDEN/REVERTED this kind of change before (negative
            # precedent with coverage) — NOT "never seen". Only then send it back, and
            # point at the specific fix you used last time. Unknown patterns are allowed
            # (observe & learn). This is the calibration thesis: act on evidence, not ignorance.
            known_bad = dec.risk >= 0.5 and dec.risk_coverage >= 0.35 and bool(dec.resolution)
            if enforce:
                # The double acts on your behalf TOWARD THE AI — it never prompts you
                # (you only talk to the AI). Known-bad -> send the AI back with your fix;
                # otherwise let it through.
                if known_bad:
                    after = dec.resolution    # the specific thing you did instead (pinpoint)
                    verdict = "deny"          # send the AI back to redo, on your behalf
                    reason = (f"[codedouble] (on the developer's behalf) they've corrected this "
                              f"pattern before in {int(dec.risk * 100)}% of similar cases — "
                              f"do this instead: {after}")
                else:
                    verdict = "allow"         # unknown or known-good -> let it through
            # DRIFT (opt-in, CODEDOUBLE_DRIFT=1): is the AI straying from THIS session's
            # established goal / constraints / decisions / avoid? Acts on the developer's
            # behalf TOWARD the AI — sends it back to re-align, never prompts the developer.
            if enforce and verdict == "allow" and os.environ.get("CODEDOUBLE_DRIFT") == "1":
                anchors = _decision_anchors(args.log, ev.get("session_id", ""))
                is_drift, dreason, dredirect = _drift_check(
                    anchors, payload["request"], payload.get("diff", ""))
                if is_drift:
                    verdict = "deny"
                    after = dredirect or "Re-align with the session goal."
                    reason = (f"[codedouble] (on the developer's behalf) this drifts from the "
                              f"session goal — {dreason}. Instead: {after}")
            # Optional quality gate (opt-in, Edit/Write only): deny ONLY when the change
            # CONTRADICTS a nameable session anchor (consistency, not sufficiency —
            # judging atomic steps against the whole goal was pure false positives).
            # No consolidated anchors yet -> nothing to contradict -> allow.
            if (enforce and verdict == "allow" and tool in ("Edit", "Write")
                    and os.environ.get("CODEDOUBLE_QC") == "1"):
                proposed = str(ti.get("new_string") or ti.get("content") or "")
                anchors_struct = _decision_anchors(args.log, ev.get("session_id", ""))
                if proposed.strip() and anchors_struct.get("goal"):
                    ok, violated, refine = _quality_check(anchors_struct, proposed)
                    if not ok:
                        verdict = "deny"
                        after = refine or "Redo it without violating that anchor."
                        reason = (f"[codedouble] quality check — this contradicts an established "
                                  f"anchor: \"{violated}\". {after}")
            _log_decision(args.log, name, tool, dec, enforce, intent=payload["request"],
                          verdict=verdict, before=before, after=after, target=target,
                          session_id=ev.get("session_id", ""), reason=reason, cwd=cwd)
            if verdict == "deny":
                _maybe_reflect(args.log)       # a send-back matters -> reflect now
            if enforce:
                print(json.dumps({"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": ("deny" if verdict == "deny" else verdict),
                    "permissionDecisionReason": reason}}))
            return  # shadow: emit nothing -> normal flow

        if name == "PostToolUse":
            # the action the double asked/denied about has completed -> capture the reply
            tool = ev.get("tool_name", "")
            ti = ev.get("tool_input", {}) or {}
            target = str(ti.get("file_path") or ti.get("command") or "")
            ran = str(ti.get("new_string") or ti.get("content") or ti.get("command") or "")
            _capture_reply(args.log, tool, target, ran)
            _session_outcome(args.log, ev.get("session_id", ""), tool, target)  # outcome loop
            _flag_bypassed_sendback(args.log, ev.get("session_id", ""), tool, target, ran)
            return
    except Exception:
        return  # fail-open: never block the user's session


def cmd_reflect(args):
    """Layered, survival-time reflection (README §6) — off the hot path. Tiers
    un-reacted changes by how long they've survived (mostly-good -> confirmed),
    then distils stable patterns into general rules. Idempotent; run on idle."""
    from .reflect import reflect_log
    raw = EventLog(args.log).read()
    if not raw:
        if not args.quiet:
            print("nothing to reflect yet")
        return
    ext = RuleBasedExtractor(HashingEmbedder(256))
    updated, rules, tiers = reflect_log(
        raw, time.time(), args.idle_fast, args.idle_good, ext, args.min_promote)
    d = os.path.dirname(args.log) or "."
    tmp = args.log + ".tmp"
    with open(tmp, "w") as f:
        for r in updated:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, args.log)
    rpath = os.path.join(d, "rules.jsonl")
    with open(rpath, "w") as f:
        for rr in rules:
            f.write(json.dumps(rr) + "\n")
    if not args.quiet:
        print("reflected (tiers):", dict(tiers))
        print(f"distilled {len(rules)} rule(s) -> {rpath}")


def cmd_models(args):
    """List local Ollama models the user can pick for the reasoning slot."""
    from .backends import list_ollama_models
    ms = list_ollama_models()
    if not ms:
        print("no local ollama models found.")
        print("  is `ollama serve` running?  then e.g.:  ollama pull qwen2.5-coder:7b")
        return
    default = resolve_ollama_model(None)
    print("local ollama models  (use:  codedouble report --backend ollama --model NAME)")
    for m in ms:
        print(f"  {'* ' if m == default else '  '}{m}")
    print(f"\ndefault when no --model/CODEDOUBLE_OLLAMA_MODEL is set:  {default}")


def cmd_setup(args):
    """Frictionless wiring: create the store, an LLM-port template, and merge the
    Claude Code hooks (intake + outtake + reply-capture) into settings.json —
    idempotent, with a backup, preserving any existing hooks."""
    import shutil
    home = codedouble_home()
    os.makedirs(home, exist_ok=True)
    print(f"[codedouble setup] store: {home}")

    lp = os.path.join(home, "llm.json")              # LLM port template (no secrets)
    if not os.path.exists(lp):
        with open(lp, "w") as f:
            json.dump({"base_url": "", "model": "",
                       "api_key_file": "", "api_key_field": []}, f, indent=2)
        print(f"  wrote LLM-port template -> {lp}  (fill base_url/model/api_key_* to enable a remote LLM)")
    else:
        print(f"  LLM port present       -> {lp}")

    sp = os.path.expanduser(args.settings)
    try:
        d = json.load(open(sp)) if os.path.exists(sp) else {}
    except Exception:
        d = {}
    if os.path.exists(sp):
        shutil.copy(sp, sp + ".codedouble-setup.bak")
    # DEFAULT is full: enforce + QC. Opt out with --shadow / --no-enforce / --no-qc.
    enforce = not (args.shadow or getattr(args, "no_enforce", False))
    qc = enforce and not (args.shadow or getattr(args, "no_qc", False))
    hooks = d.setdefault("hooks", {})
    pre = ("CODEDOUBLE_ENFORCE=1 " if enforce else "") + \
          ("CODEDOUBLE_QC=1 " if qc else "") + "python3 -m codedouble.cli hook"

    def ensure(event, matcher, command):
        arr = hooks.setdefault(event, [])
        for blk in arr:
            for h in blk.get("hooks", []):
                if "codedouble.cli hook" in h.get("command", ""):
                    h["command"] = command              # update (e.g. flag changes)
                    return "updated"
        arr.append({"matcher": matcher, "hooks": [{"type": "command", "command": command, "timeout": 30}]})
        return "added"

    r = {
        "UserPromptSubmit": ensure("UserPromptSubmit", "", "python3 -m codedouble.cli hook"),
        "PreToolUse": ensure("PreToolUse", "Edit|Write|Bash", pre),
        "PostToolUse": ensure("PostToolUse", "Edit|Write|Bash", "python3 -m codedouble.cli hook"),
    }
    try:
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        with open(sp, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  hooks -> {sp}: " + ", ".join(f"{k} {v}" for k, v in r.items()))
    except Exception as e:
        print("  could not write settings:", e)
    print(f"  mode: {'ENFORCE' if enforce else 'shadow'}{' + QC' if qc else ''}. "
          "Start a new Claude Code session to load the hooks.")


def cmd_status(args):
    raw = EventLog(args.log).read()
    hook = os.path.join(args.repo, ".git", "hooks", "post-commit")
    hooked = os.path.exists(hook) and "codedouble" in open(hook).read()
    print(f"log:    {args.log}  ({len(raw)} interactions)")
    print(f"hook:   {'ON' if hooked else 'off'}  ({hook})")
    if raw:
        from collections import Counter
        c = Counter(r.get("outcome") for r in raw)
        print("outcomes:", dict(c))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="codedouble")
    ap.add_argument("--log", default=DEFAULT_LOG)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("on", help="mine git + install hook (turn monitoring ON)")
    p.add_argument("--repo", default="."); p.set_defaults(func=cmd_on)

    p = sub.add_parser("capture-git", help="mine commits into the log")
    p.add_argument("--repo", default="."); p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_capture_git)

    p = sub.add_parser("log", help="record one interaction by hand")
    p.add_argument("request"); p.add_argument("resolution")
    p.add_argument("--outcome", default="confirmed_good")
    p.add_argument("--corrected-from", dest="corrected_from", default=None)
    p.add_argument("--lang", default=""); p.set_defaults(func=cmd_log)

    p = sub.add_parser("report", help="replay the log -> ASCII + report.html")
    p.add_argument("--backend", default="real",
                   help="real (local ST embedder) | ollama (local LLM reasoning) | mistral | default")
    p.add_argument("--model", default=None,
                   help="ollama model for --backend ollama (overrides env; see `codedouble models`)")
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--conf", type=float, default=0.6)
    p.add_argument("--out", default=os.path.join(codedouble_home(), "report.html"),
                   help="where to write the HTML (default: global ~/.codedouble/report.html)")
    p.add_argument("--sim", action="store_true", help="use the simulated user")
    p.add_argument("--seed", type=int, default=7); p.set_defaults(func=cmd_report)

    p = sub.add_parser("gate", help="gateway decision for one intent (stdin JSON) -> decision JSON")
    p.add_argument("--request", default="", help="intent text (or pipe JSON on stdin)")
    p.add_argument("--conf", type=float, default=0.6)
    p.add_argument("--shadow", action="store_true", help="never block; just log/emit the decision")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("hook", help="Claude Code hook adapter (reads hook JSON on stdin)")
    p.add_argument("--conf", type=float, default=0.6); p.set_defaults(func=cmd_hook)

    p = sub.add_parser("install-hook"); p.add_argument("--repo", default=".")
    p.set_defaults(func=lambda a: install_hook(a.repo))

    p = sub.add_parser("reflect", help="layered survival-time reflection + distil rules (run on idle)")
    p.add_argument("--idle-fast", dest="idle_fast", type=float, default=1800,
                   help="age(s) after which an un-reacted change is 'mostly good' (default 30m)")
    p.add_argument("--idle-good", dest="idle_good", type=float, default=86400,
                   help="age(s) after which it's 'confirmed good' (default 1d)")
    p.add_argument("--min-promote", dest="min_promote", type=int, default=3)
    p.add_argument("--quiet", action="store_true"); p.set_defaults(func=cmd_reflect)

    p = sub.add_parser("setup", help="frictionless: wire Claude Code hooks + store + LLM-port template")
    p.add_argument("--settings", default="~/.claude/settings.json")
    p.add_argument("--shadow", action="store_true", help="observe only (default is enforce + QC)")
    p.add_argument("--no-enforce", dest="no_enforce", action="store_true", help="don't send back; observe")
    p.add_argument("--no-qc", dest="no_qc", action="store_true", help="skip the LLM drift/quality check")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("summarize", help="consolidate each session's running notes into an intent summary (run on idle)")
    p.add_argument("--quiet", action="store_true"); p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("models", help="list local ollama models you can pick for reasoning")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("status"); p.add_argument("--repo", default=".")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help(); return
    args.func(args)


if __name__ == "__main__":
    main()
