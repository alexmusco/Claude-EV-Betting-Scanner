"""Console and HTML output."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import Config
from .db import Database
from .scan import Opportunity, ScanResult

PRETTY_MARKET = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_threes": "Threes",
    "player_blocks": "Blocks",
    "player_steals": "Steals",
    "player_turnovers": "Turnovers",
    "player_points_rebounds_assists": "PRA",
    "player_points_rebounds": "P+R",
    "player_points_assists": "P+A",
    "player_rebounds_assists": "R+A",
    "player_blocks_steals": "B+S",
    "player_pass_yds": "Pass yds",
    "player_pass_tds": "Pass TDs",
    "player_rush_yds": "Rush yds",
    "player_receptions": "Receptions",
    "player_reception_yds": "Rec yds",
    "player_shots_on_goal": "Shots on goal",
    "player_shots_on_target": "Shots on target",
    "pitcher_strikeouts": "Pitcher Ks",
    "batter_total_bases": "Total bases",
    "batter_hits": "Hits",
}


def pretty_market(key: str) -> str:
    base = key.replace("_alternate", "")
    label = PRETTY_MARKET.get(base, base.replace("player_", "").replace("_", " ").title())
    return f"{label} (alt)" if key.endswith("_alternate") else label


def describe(selection, side, line) -> str:
    """
    Render a bet the way `Opportunity.description` does, but from loose
    values so database rows and API quotes format identically.
    """
    s = (side or "").strip().lower()
    if s in ("over", "under", "yes", "no"):
        base = selection if selection and selection.strip().lower() != s else "Total"
        out = f"{base} {side}"
        return out if line is None else f"{out} {line:g}"
    if line is None:
        return f"{selection} ML"
    return f"{selection} {line:+g}"


def american(decimal: float) -> str:
    from .pricing import decimal_to_american

    a = decimal_to_american(decimal)
    return f"{a:+.0f}"


# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------


def console_table(opportunities: Sequence[Opportunity], limit: int = 40) -> str:
    if not opportunities:
        return "No opportunities cleared the thresholds."

    headers = ["#", "EV", "Liq", "Bet", "Market", "Book", "Price", "Fair", "Stake",
               "Game", "Starts", "Flags"]
    rows = []
    for i, o in enumerate(opportunities[:limit], 1):
        rows.append(
            [
                str(o.db_id if getattr(o, "db_id", None) else i),
                f"{o.ev:+.1%}",
                _liquidity_label(getattr(o, "liquidity", None)),
                o.description,
                pretty_market(o.market),
                o.soft_book,
                f"{o.soft_price:.2f} ({american(o.soft_price)})",
                f"{o.fair_price:.2f}",
                f"{o.recommended_stake:,.0f}" if o.recommended_stake else "-",
                o.matchup,
                _relative(o.commence_time, o.scanned_at),
                ",".join(o.flags) or "",
            ]
        )

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    more = ""
    if len(opportunities) > limit:
        more = f"\n\n... and {len(opportunities) - limit} more (use --limit to see them)."
    return f"{line}\n{sep}\n{body}{more}"


def _liquidity_label(score) -> str:
    """
    Coarse bucket rather than a decimal. The score is a rough confidence
    estimate built from proxies, and printing it to two places would imply
    a precision it does not have.
    """
    if score is None:
        return "-"
    if score >= 0.80:
        return "deep"
    if score >= 0.60:
        return "good"
    if score >= 0.40:
        return "thin"
    return "v.thin"


def shortlist(opportunities: Sequence[Opportunity], limit: int = 12) -> str:
    """
    The daily view: what to bet, at what price, for how much.

    Deliberately narrower than console_table. The point of a daily run is a
    list you act on in a few minutes before the prices move, so it carries
    only what you need to place the bet and drops the diagnostics.
    """
    playable = [o for o in opportunities if o.recommended_stake > 0]
    if not playable:
        return "Nothing clears the bar right now."

    out = []
    header = (
        f"{'id':>3}  {'EV':>6}  {'liq':<6} {'bet':<38} {'price':>14} "
        f"{'stake':>7}  game"
    )
    out.append(header)
    out.append("-" * len(header))
    for i, o in enumerate(playable[:limit], 1):
        price = f"{o.soft_price:.2f} ({american(o.soft_price)})"
        ident = o.db_id if getattr(o, "db_id", None) else i
        out.append(
            f"{ident:>3}  {o.ev:>+5.1%}  {_liquidity_label(o.liquidity):<6} "
            f"{o.description:<38.38} {price:>14} "
            f"{o.recommended_stake:>7,.0f}  "
            f"{o.matchup[:40]} ({_relative(o.commence_time, o.scanned_at)})"
        )
    total = sum(o.recommended_stake for o in playable[:limit])
    out.append("")
    out.append(
        f"{len(playable)} playable, showing {min(limit, len(playable))}. "
        f"Total recommended stake {total:,.0f}."
    )
    out.append("The id column is what `betedge bet <id> --stake <amount>` takes.")
    return "\n".join(out)


def scan_summary(result: ScanResult) -> str:
    took = (result.finished_at - result.started_at).total_seconds()
    lines = [
        f"Scanned {result.events_scanned} events across {', '.join(result.sports)} in {took:.0f}s",
        f"{result.quotes_seen:,} quotes seen, {result.sharp_markets_paired:,} two-sided sharp markets",
        f"{len(result.clean)} flagged, {len(result.suspect)} suspect",
        f"Credits: {result.credits_spent} spent this run"
        + (f", {result.credits_remaining:,} remaining" if result.credits_remaining is not None else ""),
    ]
    if result.rejections:
        top = sorted(result.rejections.items(), key=lambda kv: -kv[1])[:6]
        lines.append("Filtered out: " + ", ".join(f"{k} ({v:,})" for k, v in top))
    if result.errors:
        lines.append(f"{len(result.errors)} error(s): " + " | ".join(result.errors[:3]))
    return "\n".join(lines)


def _relative(when: datetime, now: datetime) -> str:
    mins = (when - now).total_seconds() / 60
    if mins < 60:
        return f"{mins:.0f}m"
    if mins < 60 * 24:
        return f"{mins/60:.1f}h"
    return f"{mins/1440:.1f}d"


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root{--bg:#fbfaf9;--panel:#fff;--ink:#1c1917;--muted:#78716c;--line:#e7e5e4;
--pos:#15803d;--pos-bg:#dcfce7;--warn:#b45309;--warn-bg:#fef3c7;--accent:#1d4ed8}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0c0a09;
--panel:#1c1917;--ink:#f5f5f4;--muted:#a8a29e;--line:#292524;--pos:#4ade80;
--pos-bg:#14321f;--warn:#fbbf24;--warn-bg:#3a2a08;--accent:#93c5fd}}
:root[data-theme=dark]{--bg:#0c0a09;--panel:#1c1917;--ink:#f5f5f4;--muted:#a8a29e;
--line:#292524;--pos:#4ade80;--pos-bg:#14321f;--warn:#fbbf24;--warn-bg:#3a2a08;--accent:#93c5fd}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,-apple-system,
"Segoe UI",Roboto,sans-serif;margin:0;padding-block:32px;padding-left:20px;padding-right:20px}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:16px;margin:32px 0 10px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px;margin:0 0 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:20px;font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{text-align:left;font-weight:600;color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.05em;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
.num{font-variant-numeric:tabular-nums;text-align:right}
.ev{font-weight:600;color:var(--pos);background:var(--pos-bg);border-radius:5px;
padding:2px 7px;font-variant-numeric:tabular-nums}
.bet{font-weight:600}
.dim{color:var(--muted)}
.flag{display:inline-block;background:var(--warn-bg);color:var(--warn);border-radius:5px;
padding:1px 6px;font-size:11px;margin-right:4px}
.note{color:var(--muted);font-size:12.5px;margin-top:10px;line-height:1.6}
.empty{padding:28px;text-align:center;color:var(--muted)}
"""


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def scan_report_html(result: ScanResult, cfg: Config, title: str = "Scan") -> str:
    def row(o: Opportunity, idx: int) -> str:
        flags = "".join(f'<span class="flag">{_esc(f)}</span>' for f in o.flags)
        stake = f"{o.recommended_stake:,.0f}" if o.recommended_stake else '<span class="dim">—</span>'
        return f"""<tr>
<td class="num dim">{idx}</td>
<td><span class="ev">{o.ev:+.1%}</span></td>
<td class="bet">{_esc(o.description)}</td>
<td class="dim">{_esc(pretty_market(o.market))}</td>
<td>{_esc(o.soft_book)}</td>
<td class="num">{o.soft_price:.2f} <span class="dim">{american(o.soft_price)}</span></td>
<td class="num dim">{o.fair_price:.2f}</td>
<td class="num">{stake}</td>
<td class="dim">{_esc(o.matchup)}</td>
<td class="num dim">{_relative(o.commence_time, o.scanned_at)}</td>
<td>{flags}</td>
</tr>"""

    def table(items: Sequence[Opportunity], empty: str) -> str:
        if not items:
            return f'<div class="scroll"><div class="empty">{_esc(empty)}</div></div>'
        body = "".join(row(o, i) for i, o in enumerate(items, 1))
        return f"""<div class="scroll"><table>
<thead><tr><th></th><th>EV</th><th>Bet</th><th>Market</th><th>Book</th>
<th class="num">Price</th><th class="num">Fair</th><th class="num">Stake</th>
<th>Game</th><th class="num">Starts</th><th>Flags</th></tr></thead>
<tbody>{body}</tbody></table></div>"""

    stamp = result.finished_at.astimezone().strftime("%a %d %b %Y, %H:%M %Z")
    total_stake = sum(o.recommended_stake for o in result.clean)
    cards = [
        ("Flagged", str(len(result.clean))),
        ("Suspect", str(len(result.suspect))),
        ("Events", f"{result.events_scanned:,}"),
        ("Quotes", f"{result.quotes_seen:,}"),
        ("Credits used", f"{result.credits_spent:,}"),
        ("Credits left", f"{result.credits_remaining:,}" if result.credits_remaining is not None else "—"),
        ("Total stake", f"{total_stake:,.0f}"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v in cards
    )

    rejected = ""
    if result.rejections:
        items = sorted(result.rejections.items(), key=lambda kv: -kv[1])
        rejected = "<br>".join(f"{_esc(k.replace('_',' '))}: {v:,}" for k, v in items)
        rejected = f'<h2>Why quotes were filtered out</h2><p class="note">{rejected}</p>'

    return f"""<title>Bet Scan {stamp}</title>
<style>{_CSS}</style>
<div class="wrap">
<h1>{_esc(title)}</h1>
<p class="sub">{_esc(stamp)} &middot; {_esc(', '.join(result.sports))} &middot;
sharp book {_esc(cfg.books.sharp)} vs {_esc(', '.join(cfg.books.soft))} &middot;
de-vig {_esc(cfg.model.devig_method)} &middot; threshold {cfg.model.min_ev:+.1%} EV</p>
<div class="cards">{card_html}</div>
<h2>Playable</h2>
{table(result.clean, "Nothing cleared the thresholds this run. That is a normal result.")}
<p class="note">Fair price is the sharp book's price with its margin stripped out.
Stake is {cfg.bankroll.kelly_multiplier:g}x Kelly on a {cfg.bankroll.amount:,.0f} bankroll,
capped at {cfg.bankroll.max_fraction:.0%} per bet. Prices move; re-check before betting.</p>
<h2>Suspect</h2>
{table(result.suspect, "Nothing flagged as suspect.")}
<p class="note">These tripped a sanity check &mdash; an edge too large to believe, a stale
quote, or an edge that only exists under some de-vig methods. Usually a line about to be
pulled or a mismatched selection rather than free money. No stake is recommended.</p>
{rejected}
</div>"""


def performance_report_html(db: Database, cfg: Config) -> str:
    s = db.summary()

    def fmt(v, kind="num"):
        if v is None:
            return "—"
        if kind == "pct":
            return f"{v:+.1%}"
        if kind == "pct0":
            return f"{v:.0%}"
        if kind == "money":
            return f"{v:+,.2f}"
        return f"{v:,.0f}"

    cards = [
        ("Settled bets", fmt(s["bets_settled"])),
        ("Pending", fmt(s["bets_pending"])),
        ("Staked", f"{s['total_staked']:,.0f}"),
        ("P&L", fmt(s["total_pnl"], "money")),
        ("ROI", fmt(s["roi"], "pct")),
        ("Win rate", fmt(s["win_rate"], "pct0")),
        ("Avg CLV", fmt(s["avg_clv"], "pct")),
        ("CLV beat rate", fmt(s["clv_positive_rate"], "pct0")),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v in cards
    )

    def group_table(column: str, label: str) -> str:
        rows = db.breakdown(column)
        if not rows:
            return ""
        body = "".join(
            f"<tr><td>{_esc(r['key'])}</td><td class='num'>{r['n']}</td>"
            f"<td class='num'>{r['staked']:,.0f}</td>"
            f"<td class='num'>{(r['pnl'] or 0):+,.2f}</td>"
            f"<td class='num'>{fmt(r['roi'],'pct')}</td>"
            f"<td class='num dim'>{fmt(r['avg_ev'],'pct')}</td></tr>"
            for r in rows
        )
        return f"""<h2>By {_esc(label)}</h2><div class="scroll"><table>
<thead><tr><th>{_esc(label)}</th><th class="num">Bets</th><th class="num">Staked</th>
<th class="num">P&amp;L</th><th class="num">ROI</th><th class="num">Avg EV</th></tr></thead>
<tbody>{body}</tbody></table></div>"""

    open_rows = db.open_bets()
    if open_rows:
        body = "".join(
            f"<tr><td class='num dim'>{r['id']}</td><td class='bet'>{_esc(r['selection'])} "
            f"{_esc(r['side'])} {_esc(r['line'] if r['line'] is not None else '')}</td>"
            f"<td class='dim'>{_esc(pretty_market(r['market'] or ''))}</td>"
            f"<td>{_esc(r['book'])}</td><td class='num'>{r['price']:.2f}</td>"
            f"<td class='num'>{r['stake']:,.0f}</td>"
            f"<td class='num'>{fmt(r['ev_at_bet'],'pct')}</td>"
            f"<td class='dim'>{_esc(r['matchup'])}</td></tr>"
            for r in open_rows
        )
        open_html = f"""<h2>Open bets</h2><div class="scroll"><table>
<thead><tr><th>ID</th><th>Bet</th><th>Market</th><th>Book</th><th class="num">Price</th>
<th class="num">Stake</th><th class="num">EV</th><th>Game</th></tr></thead>
<tbody>{body}</tbody></table></div>"""
    else:
        open_html = '<h2>Open bets</h2><div class="scroll"><div class="empty">None.</div></div>'

    clv_note = (
        f"Closing-line value is measured on {s['clv_sample']} bets. "
        "It is the better early signal: if your average CLV is positive, the model is "
        "finding genuine mispricing even when the P&amp;L is negative. If CLV is around "
        "zero but P&amp;L is positive, you have been lucky, not right."
        if s["clv_sample"]
        else "No closing lines captured yet. Run <code>betedge close</code> shortly before "
        "each event starts to build this up &mdash; it is the fastest way to tell whether "
        "the model works."
    )

    expected = (
        f"<p class='note'>Expected P&amp;L from the EV at bet time was "
        f"{s['expected_pnl']:+,.2f}; actual is {s['total_pnl']:+,.2f}, a gap of "
        f"{s['pnl_vs_expected']:+,.2f}. Over a small sample this gap is almost entirely "
        f"variance.</p>"
        if s["pnl_vs_expected"] is not None
        else ""
    )

    stamp = datetime.now(timezone.utc).astimezone().strftime("%a %d %b %Y, %H:%M %Z")
    return f"""<title>Betting Performance</title>
<style>{_CSS}</style>
<div class="wrap">
<h1>Betting performance</h1>
<p class="sub">{_esc(stamp)} &middot; bankroll {cfg.bankroll.amount:,.0f}</p>
<div class="cards">{card_html}</div>
{expected}
<p class="note">{clv_note}</p>
{open_html}
{group_table("sport", "sport")}
{group_table("market", "market")}
{group_table("book", "book")}
</div>"""


_SKELETON = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head><body>
{content}
</body></html>
"""


def write_report(path: str | Path, content: str, standalone: bool = True) -> Path:
    """
    Write a report to disk.

    The report builders return a body fragment (title, style, markup) with no
    document skeleton, so the same string can be handed to a hosting tool
    that supplies its own. `standalone=True` wraps it for opening as a local
    file in a browser.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        _SKELETON.format(content=content) if standalone else content, encoding="utf-8"
    )
    return p


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def distribution_report(assessments: Sequence, min_ev: float, top: int = 15) -> str:
    """
    What the whole board looked like, not just what cleared the bar.

    A scan that flags nothing tells you a bar was not met; it does not tell
    you by how much. Whether the best quote on the board sat at -0.3% or at
    -6% is the difference between a threshold set slightly too high and a
    market with no edge in it at all, and those call for opposite responses.
    """
    if not assessments:
        return ("Nothing was priced. Either no event was in window, or the "
                "sharp book had no complete two-sided market to compare.")

    evs = [a.ev for a in assessments]
    out: list[str] = []
    books = sorted({a.book for a in assessments})
    out.append(f"{len(assessments):,} soft-book quotes priced "
               f"({', '.join(books)}) against the de-vigged sharp price.")
    out.append("")

    # ---- EV distribution ------------------------------------------------
    out.append("EXPECTED VALUE ACROSS THE BOARD")
    for label, q in [("worst", 0.0), ("25th", 0.25), ("median", 0.5),
                     ("75th", 0.75), ("90th", 0.90), ("best", 1.0)]:
        out.append(f"  {label:<8} {_percentile(evs, q):+7.2%}")
    positive = [e for e in evs if e > 0]
    out.append(f"  {len(positive):,} of {len(evs):,} priced above zero "
               f"({len(positive)/len(evs):.0%})")
    out.append("")

    # ---- where the bar sits --------------------------------------------
    out.append("WHAT DIFFERENT BARS WOULD HAVE FLAGGED")
    for bar in (0.04, 0.03, 0.02, 0.01, 0.005, 0.0):
        n = sum(1 for a in assessments if a.ev >= bar)
        marker = "  <- your min_ev" if abs(bar - min_ev) < 1e-9 else ""
        out.append(f"  at {bar:+.1%}: {n:3d} quote(s){marker}")
    cleared = sum(1 for a in assessments if a.ev >= a.required_ev)
    out.append(f"  after the liquidity adjustment: {cleared} flagged")
    out.append("")

    # ---- sharp book margin, the liquidity input ------------------------
    out.append("PINNACLE OVERROUND BY MARKET TIER  (the liquidity signal)")
    by_tier: dict[str, list] = {}
    for a in assessments:
        by_tier.setdefault(a.tier, []).append(a)
    for tier, rows in sorted(by_tier.items(), key=lambda kv: -len(kv[1])):
        orr = [r.overround for r in rows]
        liq = [r.liquidity for r in rows]
        out.append(
            f"  {tier:<16} n={len(rows):<4} overround "
            f"{_percentile(orr, 0.25):.2%} / {_percentile(orr, 0.5):.2%} / "
            f"{_percentile(orr, 0.75):.2%}   liquidity median "
            f"{_percentile(liq, 0.5):.2f}"
        )
    out.append("  (quartiles: 25th / median / 75th)")
    out.append("")

    # ---- the near misses ------------------------------------------------
    out.append(f"CLOSEST {top} TO CLEARING, BEST FIRST")
    header = (f"  {'EV':>7} {'bar':>7} {'short':>7}  {'liq':>5} {'orr':>6}  "
              f"{'bet':<34} {'price':>6}  game")
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    for a in sorted(assessments, key=lambda x: x.shortfall)[:top]:
        out.append(
            f"  {a.ev:>+6.2%} {a.required_ev:>+6.2%} {a.shortfall:>+6.2%}  "
            f"{a.liquidity:>5.2f} {a.overround:>5.2%}  "
            f"{a.description:<34.34} {a.soft_price:>6.2f}  {a.matchup[:30]}"
        )
    return "\n".join(out)
