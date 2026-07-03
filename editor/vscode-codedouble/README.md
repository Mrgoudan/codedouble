# CodeDouble — your double, watching your AI

This panel is the visible face of **[CodeDouble](https://gitee.com/ziruichen12138/codedouble)** —
an external monitor that sits beside your AI coding session and acts on your behalf, toward the
AI, without ever prompting you.

## What the panel shows

- **This session's goal and anchors** — the intent (goal / constraints / decisions / todos /
  avoid-list) the monitor maintains from your conversation and injects to steer the AI each turn.
- **Send-backs, with the receipt** — every change the monitor rejected: what the AI tried → which
  of your anchors it violated, verbatim, and how it was told to redo it.
- **Counts** — sent back vs let through, for the AI session running in this window's folder.

It also captures editor reactions (accept / override / reject) passively — no labelling — as
learning signal for the monitor.

## Requires the CodeDouble gateway

The extension alone is the dashboard. The monitor itself (gateway hooks + memory + gates) is a
small Python package wired into your AI harness:

```bash
git clone https://gitee.com/ziruichen12138/codedouble.git && cd codedouble && ./setup.sh
```

`setup.sh` is idempotent: installs the package, pulls the local model (optional), wires the
hooks, and builds this panel. Use `./setup.sh --shadow` for observe-only mode.

## Tips

- The panel lives in its own Activity Bar icon (the eye). On profiles with very many extensions,
  VS Code can race view registration and drop it into Explorer — drag the "This session" view
  onto the Activity Bar once to pin it permanently.
- The panel is read-only by design: the monitor learns from what you do, never from manual
  memory curation.

MIT licensed. Issues and design record: [gitee.com/ziruichen12138/codedouble](https://gitee.com/ziruichen12138/codedouble).
