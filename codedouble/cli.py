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


def _log_decision(log_path, event, tool, dec, enforce, intent="", verdict="", before="", after="", target=""):
    d = os.path.dirname(log_path) or "."
    path = os.path.join(d, "decisions.jsonl")
    try:
        os.makedirs(d, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({
                "ts": time.time(), "event": event, "tool": tool,
                "target": (target or "").strip()[:160],   # file / command (for reply correlation)
                "intent": (intent or "").strip()[:140],
                "resolution": (dec.resolution or "").strip()[:140],
                "before": (before or "").strip()[:200],   # what the AI proposed
                "after": (after or "").strip()[:200],      # the correction the double asked for
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
    "You are a strict reviewer of an AI's proposed code change. Given the developer's "
    "intent and the proposed change, decide if it adequately and correctly fulfills the "
    "intent. Return ONLY JSON: {\"ok\": true} if it is good enough, or "
    "{\"ok\": false, \"refine\": \"<one concise, specific instruction telling the AI how to "
    "redo it better>\"} if it is not."
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


def _quality_check(intent, proposed):
    """Ask the LOCAL model whether the change meets the intent. Returns
    (ok, refine). Fail-open: any error -> ok=True (never block on QC failure)."""
    try:
        from .backends import llm_complete                       # frequent / per-edit -> local first
        out = llm_complete(f"Intent:\n{intent}\n\nProposed change:\n{proposed[:2000]}",
                           system=_QC_SYSTEM, json_mode=True, timeout=40, prefer="local")
        data = json.loads(out)
        return bool(data.get("ok", True)), str(data.get("refine", ""))[:200]
    except Exception:
        return True, ""


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

# a decision being made is also a moment that matters -> re-anchor
_DECISION_RE = re.compile(
    r"\b(let'?s\b|go with|we'?ll\b|we will\b|decided|the plan is|stick with|going with|finali[sz]e)\b", re.I)


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


def _session_note(log_path, sid, prompt):
    """Incrementally append the user's intent to this session's running notes."""
    try:
        with open(_sid_file(log_path, sid, ".jsonl"), "a") as f:
            f.write(json.dumps({"ts": time.time(), "prompt": (prompt or "")[:500]}) + "\n")
    except Exception:
        pass


_ANCHORS_SYSTEM = (
    "Extract the durable ANCHORS of this coding session from the developer's prompts. "
    "Return ONLY JSON: {\"goal\": \"<the end goal in one sentence>\", "
    "\"constraints\": [hard requirements that must hold], "
    "\"decisions\": [decisions already made — do NOT re-litigate], "
    "\"todos\": [open tasks still to do], \"avoid\": [approaches they rejected]}. "
    "Each item one short line; omit empties."
)


def _consolidate(notes):
    """Distil running notes into structured ANCHORS (JSON). Local LLM if available;
    a minimal extractive anchor set otherwise."""
    text = "\n".join(f"- {n}" for n in notes[-50:])
    try:
        from .backends import llm_complete                       # HARD, session-wide -> remote (GLM)
        out = llm_complete(text, system=_ANCHORS_SYSTEM, json_mode=True, timeout=90, prefer="remote")
        json.loads(out)                       # validate
        return out
    except Exception:
        return json.dumps({"goal": "", "constraints": [], "decisions": [],
                           "todos": [n[:160] for n in notes[-8:]], "avoid": []})


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


def _global_anchors_path(log_path):
    return os.path.join(os.path.dirname(log_path) or ".", "anchors.global.json")


def _merge_global(log_path, a):
    """Graduate a session's durable anchors (constraints / decisions / avoid) into
    the cross-session GLOBAL anchors — the local→over-time bridge. Union+dedupe
    (an LLM dedupe could refine this later)."""
    if not isinstance(a, dict):
        return
    gp = _global_anchors_path(log_path)
    try:
        g = json.loads(open(gp).read()) if os.path.exists(gp) else {}
    except Exception:
        g = {}
    for key in ("constraints", "decisions", "avoid"):
        seen = {str(x).strip().lower() for x in (g.get(key) or [])}
        merged = list(g.get(key) or [])
        for x in (a.get(key) or []):
            s = str(x).strip()
            if s and s.lower() not in seen:
                merged.append(x); seen.add(s.lower())
        g[key] = merged[-40:]
    if a.get("goal") and not g.get("goal"):
        g["goal"] = a["goal"]
    try:
        with open(gp, "w") as f:
            f.write(json.dumps(g))
    except Exception:
        pass


def _session_anchors(log_path, sid):
    """Anchors (rendered) for steering. The session's own anchors if consolidated;
    else SEED from the durable global anchors so a fresh session starts primed
    (over-time→local), not blank."""
    try:
        ap = _sid_file(log_path, sid, ".anchors.json")
        if os.path.exists(ap):
            return _render_anchors(json.loads(open(ap).read()))
    except Exception:
        pass
    try:
        gp = _global_anchors_path(log_path)
        if os.path.exists(gp):
            return _render_anchors(json.loads(open(gp).read()))
    except Exception:
        pass
    return ""


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
    """Consolidate each session's running notes into a compact intent summary
    (local LLM if available; extractive fallback). Run on idle. Idempotent."""
    import glob
    n = 0
    for notes_path in glob.glob(os.path.join(_session_dir(args.log), "*.jsonl")):
        try:
            notes = [json.loads(l).get("prompt", "") for l in open(notes_path) if l.strip()]
        except Exception:
            continue
        if not notes:
            continue
        anchors_json = _consolidate(notes)
        try:
            with open(notes_path[:-6] + ".anchors.json", "w") as f:
                f.write(anchors_json)
            _merge_global(args.log, json.loads(anchors_json))     # graduate -> global (over-time)
            n += 1
        except Exception:
            pass
    if not args.quiet:
        print(f"summarized {n} session(s)")


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

        if name == "UserPromptSubmit":
            prompt = ev.get("prompt", "")
            _capture_correction(args.log, prompt, ev.get("transcript_path"))   # learn from "no, I meant X"
            _session_note(args.log, ev.get("session_id", ""), prompt)          # incremental session summary
            dec = _decide(args.log, {"request": prompt, "reversibility": "low"}, args.conf)
            injected = bool(dec.retrieved and dec.confidence >= 0.4 and dec.resolution)
            _log_decision(args.log, name, None, dec, enforce, intent=prompt,
                          verdict=("inject" if injected else "watch"))
            if _DECISION_RE.search(prompt):
                _maybe_reflect(args.log, summarize=True)      # a decision was made -> re-anchor
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
            # Optional quality gate (opt-in, Edit/Write only): if the proposed change
            # doesn't meet the intent, reject + paraphrase + send it back to redo.
            if (enforce and verdict == "allow" and tool in ("Edit", "Write")
                    and os.environ.get("CODEDOUBLE_QC") == "1"):
                proposed = str(ti.get("new_string") or ti.get("content") or "")
                if proposed.strip():
                    # check against the SESSION SUMMARY (established intent), not just the last prompt
                    intent_ref = (_session_summary(args.log, ev.get("session_id", ""))
                                  or _last_prompt(args.log) or payload["request"])
                    ok, refine = _quality_check(intent_ref, proposed)
                    if not ok:
                        verdict = "deny"
                        after = refine or "Redo it to match what was asked, more carefully."
                        reason = f"[codedouble] quality check — this doesn't fully meet the intent. {after}"
            _log_decision(args.log, name, tool, dec, enforce, intent=payload["request"],
                          verdict=verdict, before=before, after=after, target=target)
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
