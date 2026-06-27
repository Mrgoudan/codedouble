"""Visualization — dependency-free (terminal sparkline + standalone HTML/SVG).

Shows the effect: the §8 override-rate on confident-silent decisions, the
ask-rate, and accuracy over time. No matplotlib needed.
"""

from __future__ import annotations

from typing import List, Optional

from .metrics import Record, window_stats

_SPARK = " ▁▂▃▄▅▆▇█"


def _spark(vals: List[Optional[float]]) -> str:
    out = ""
    for v in vals:
        if v is None:
            out += "·"
        else:
            out += _SPARK[min(8, max(0, int(round(v * 8))))]
    return out


def ascii_report(records: List[Record], window: int = 40, conf_threshold: float = 0.6) -> List[dict]:
    rows = window_stats(records, window, conf_threshold)
    print(f"{'win':>3} {'§8':>6} {'ask':>6} {'acc':>6} {'n':>5}")
    for r in rows:
        s8 = "  n/a" if r["s8"] is None else f"{r['s8']:.2f}"
        print(f"{r['i']:>3} {s8:>6} {r['ask']:>6.2f} {r['acc']:>6.2f} {r['n']:>5}")
    print("§8  " + _spark([r["s8"] for r in rows]) + "   (override-rate on confident-silent — want it low/falling)")
    print("acc " + _spark([r["acc"] for r in rows]) + "   (accuracy — want it high/rising)")
    return rows


def render_html(
    records: List[Record],
    path: str = "report.html",
    window: int = 40,
    title: str = "codedouble — effect",
    conf_threshold: float = 0.6,
    subtitle: str = "",
) -> str:
    rows = window_stats(records, window, conf_threshold)
    W, H, pad = 760, 320, 48
    n = max(1, len(rows) - 1)

    def X(i: int) -> float:
        return pad + (W - 2 * pad) * (i / n)

    def Y(v: float) -> float:
        return H - pad - (H - 2 * pad) * v

    def poly(key: str) -> str:
        return " ".join(
            f"{X(r['i']):.1f},{Y(r[key]):.1f}" for r in rows if r[key] is not None
        )

    grid = "".join(
        f'<line x1="{pad}" y1="{Y(v):.1f}" x2="{W-pad}" y2="{Y(v):.1f}" stroke="#eee"/>'
        f'<text x="8" y="{Y(v)+4:.1f}" font-size="11" fill="#999">{v:.1f}</text>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    series = [
        ("s8", "#d9534f", "§8 override-rate (confident-silent)"),
        ("ask", "#5bc0de", "ask-rate"),
        ("acc", "#5cb85c", "accuracy"),
    ]
    lines = "".join(
        f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{poly(k)}"/>'
        for k, c, _ in series
    )
    dots = "".join(
        f'<circle cx="{X(r["i"]):.1f}" cy="{Y(r[k]):.1f}" r="2.5" fill="{c}"/>'
        for k, c, _ in series for r in rows if r[k] is not None
    )
    legend = "".join(
        f'<rect x="{pad + i*250}" y="8" width="12" height="12" fill="{c}"/>'
        f'<text x="{pad + i*250 + 18}" y="19" font-size="12" fill="#333">{lab}</text>'
        for i, (k, c, lab) in enumerate(series)
    )
    n_events = len(records)
    final = rows[-1] if rows else {"s8": None, "ask": 0.0, "acc": 0.0}
    fs8 = "n/a" if final["s8"] is None else f"{final['s8']:.2f}"
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<body style="font-family:system-ui,Arial;margin:24px;color:#222">
<h2>{title}</h2>
<p style="color:#666">{subtitle}{n_events} events &middot; window={window} &middot;
final: §8={fs8}, ask={final['ask']:.2f}, acc={final['acc']:.2f}</p>
<svg width="{W}" height="{H}" style="border:1px solid #ddd;background:#fff">
{grid}{legend}{lines}{dots}
<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#bbb"/>
<text x="{W//2}" y="{H-12}" font-size="11" fill="#999" text-anchor="middle">time (windows of {window} interactions) &rarr;</text>
</svg>
<p style="color:#666;max-width:760px;font-size:13px">Read: as the index fills, the
double should <b>ask less</b> and its <b>confident-silent</b> calls should stay
right — the red §8 line falls toward the irreducible floor while green accuracy
holds. On a <i>real</i> log this is the make-or-break: it only bends if your real
behavioral signatures actually cluster.</p>
</body>"""
    with open(path, "w") as f:
        f.write(html)
    return path
