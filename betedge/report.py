"""Console and HTML output."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import Config
from .db import Database
from .markets import PRETTY_MARKET, pretty_market  # noqa: F401  (re-export)
from .scan import Opportunity, ScanResult, _render_bet

def describe(selection, side, line, market=None) -> str:
    """
    Render a bet from loose values, so a database row and a live quote
    format identically. Delegates to the one renderer in scan.py rather
    than reimplementing it -- the two copies had already drifted, and the
    drift was the market name going missing.
    """
    return _render_bet(selection, side, line, market)


def american(decimal: float) -> str:
    """The price as a sportsbook shows it. 2.22 -> '+122'."""
    from .pricing import format_american

    return format_american(decimal)


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
                f"{american(o.soft_price)} ({o.soft_price:.2f})",
                f"{american(o.fair_price)}",
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
        f"{'id':>3}  {'EV':>6}  {'liq':<6} {'bet':<44} {'price':>14} "
        f"{'stake':>7}  game"
    )
    out.append(header)
    out.append("-" * len(header))
    for i, o in enumerate(playable[:limit], 1):
        price = f"{american(o.soft_price)} ({o.soft_price:.2f})"
        ident = o.db_id if getattr(o, "db_id", None) else i
        out.append(
            f"{ident:>3}  {o.ev:>+5.1%}  {_liquidity_label(o.liquidity):<6} "
            f"{o.description:<44.44} {price:>14} "
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
            f"<td>{_esc(r['book'])}</td>"
            f"<td class='num'>{_esc(american(r['price']))}</td>"
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

    # Multi-leg entries are reported separately rather than pooled in with
    # single bets. Their variance is an order of magnitude higher, so a
    # combined ROI is dominated by whichever kind happened to run hot and
    # tells you nothing about either.
    p = db.parlay_summary()
    if p["tickets_generated"]:
        parlay_cards = "".join(
            f'<div class="card"><div class="k">{_esc(k)}</div>'
            f'<div class="v">{_esc(v)}</div></div>'
            for k, v in [
                ("Tickets generated", fmt(p["tickets_generated"])),
                ("Entries settled", fmt(p["entries_settled"])),
                ("Entries pending", fmt(p["entries_pending"])),
                ("Staked", f"{p['total_staked']:,.0f}"),
                ("P&L", fmt(p["total_pnl"], "money")),
                ("ROI", fmt(p["roi"], "pct")),
                ("Avg ticket CLV", fmt(p["avg_ticket_clv"], "pct")),
            ]
        )
        parlay_note = (
            f"Ticket closing-line value is measured on {p['clv_sample']} ticket(s). "
            "It is the only fast read on whether the multi-leg model works: profit "
            "and loss on parlays is so noisy that a hundred settled entries still "
            "say nothing, whereas re-pricing every leg at the close and recomputing "
            "the joint probability gives a comparable number on every ticket, bet "
            "or not."
            if p["clv_sample"]
            else "No ticket closing lines captured yet. Run <code>betedge close</code> "
            "before each game starts &mdash; tickets that were never bet are captured "
            "too, and they are most of the evidence."
        )
        parlay_html = (
            f'<h2>Multi-leg tickets</h2><div class="cards">{parlay_cards}</div>'
            f'<p class="note">{parlay_note}</p>'
        )
    else:
        parlay_html = ""

    # The two strategies side by side, on identical metrics. Placed above
    # the per-strategy detail because "which of these is working" is the
    # question the whole log exists to answer.
    comparison = db.compare_strategies()
    comp_rows = []
    for label, getter, kind in [
        ("Settled bets", lambda x: x.settled, "int"),
        ("Staked", lambda x: x.staked, "plain"),
        ("P&amp;L", lambda x: x.pnl, "money"),
        ("ROI", lambda x: x.roi, "pct"),
        ("Modelled P&amp;L", lambda x: x.modelled_pnl, "money"),
        ("Realised / modelled", lambda x: x.realisation, "ratio"),
        ("Avg CLV", lambda x: x.avg_clv, "pct"),
        ("CLV sample", lambda x: len(x.clv_values), "int"),
    ]:
        cells = []
        for strategy in comparison.strategies:
            value = getter(strategy)
            if value is None:
                cells.append("&mdash;")
            elif kind == "pct":
                cells.append(f"{value:+.2%}")
            elif kind == "money":
                cells.append(f"{value:+,.2f}")
            elif kind == "ratio":
                cells.append(f"&times;{value:,.2f}")
            elif kind == "int":
                cells.append(f"{value:,}")
            else:
                cells.append(f"{value:,.0f}")
        comp_rows.append(
            f"<tr><td>{label}</td>"
            + "".join(f"<td class='num'>{c}</td>" for c in cells)
            + "</tr>"
        )

    interval_cells = []
    for strategy in comparison.strategies:
        bounds = strategy.roi_interval()
        interval_cells.append(
            "&mdash;" if bounds is None
            else f"{bounds[0]:+.1%} to {bounds[1]:+.1%}"
        )
    comp_rows.insert(
        4,
        f"<tr><td>ROI {comparison.confidence:.0%} interval</td>"
        + "".join(f"<td class='num dim'>{c}</td>" for c in interval_cells)
        + "</tr>",
    )

    comp_head = "".join(
        f"<th class='num'>{_esc(s.name)}</th>" for s in comparison.strategies
    )
    comparison_html = f"""<h2>Which strategy is working</h2>
<div class="scroll"><table><thead><tr><th></th>{comp_head}</tr></thead>
<tbody>{''.join(comp_rows)}</tbody></table></div>
<p class="note">{_esc(comparison.verdict())}</p>
<p class="note">Read the closing-line rows before the profit rows. CLV
converges in dozens of bets where profit needs thousands, and
&ldquo;realised over modelled&rdquo; asks whether each strategy's claimed
edge actually showed up &mdash; which is a different question from which
one happened to win more.</p>"""

    stamp = datetime.now(timezone.utc).astimezone().strftime("%a %d %b %Y, %H:%M %Z")
    return f"""<title>Betting Performance</title>
<style>{_CSS}</style>
<div class="wrap">
<h1>Betting performance</h1>
<p class="sub">{_esc(stamp)} &middot; bankroll {cfg.bankroll.amount:,.0f}</p>
<div class="cards">{card_html}</div>
{expected}
<p class="note">{clv_note}</p>
{comparison_html}
{open_html}
{parlay_html}
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
# Multi-leg tickets
#
# Every view here puts correlated EV next to independent EV, because that
# comparison is the one thing a reader must not be able to miss. A ticket
# that is positive only under an assumed correlation is a hypothesis, and
# a report that showed only the correlated number would present it as a
# finding.
# --------------------------------------------------------------------------


def parlay_console(tickets: Sequence, limit: int = 12, title: str = "") -> str:
    """One line per ticket, with the legs indented beneath it."""
    if not tickets:
        return "No tickets cleared the thresholds."

    out = []
    if title:
        out.append(title)
    header = (
        f"{'id':>4}  {'EV':>7}  {'indep':>7}  {'P(all)':>7}  {'mult':>6}  "
        f"{'stake':>6}  legs"
    )
    out.append(header)
    out.append("-" * len(header))

    for i, t in enumerate(tickets[:limit], 1):
        ident = t.db_id if getattr(t, "db_id", None) else i
        stake = f"{t.recommended_stake:,.0f}" if t.recommended_stake else "-"
        out.append(
            f"{ident:>4}  {t.ev:>+6.1%}  {t.ev_independent:>+6.1%}  "
            f"{t.joint_prob:>6.1%}  {t.payout_all_hit:>5.1f}x  {stake:>6}  "
            f"{t.n_legs}-leg {t.product.title}"
        )
        for leg in t.legs:
            out.append(
                f"        {leg.description:<44.44} "
                f"p={leg.hit_prob:>5.1%}  {pretty_market(leg.market):<16.16} "
                f"{leg.matchup[:34]}"
            )
        out.append(
            f"        correlation: {t.correlation.summary()}"
            + (f"  |  {', '.join(t.flags[:3])}" if t.flags else "")
        )
    if len(tickets) > limit:
        out.append(f"\n... and {len(tickets) - limit} more.")
    return "\n".join(out)


def roster_note(result, cfg: Config) -> str:
    """
    One line on how much of the slate had a known club.

    Worth printing on every scan, because the cost of missing rosters is
    invisible otherwise: every leg without a team silently drops from the
    sharp same-team prior to the weaker blended one, and the EV just comes
    out a little lower with nothing to say why.
    """
    book = getattr(result, "rosters", None)
    lines = []
    refresh = getattr(result, "roster_refresh", None)
    if refresh is not None:
        for sport, n in sorted(refresh.refreshed.items()):
            lines.append(f"Rosters: refreshed {n:,} players for {sport}.")
        for error in refresh.errors:
            lines.append(f"Rosters: {error}")

    legs = [leg for t in result.tickets for leg in t.legs]
    if legs:
        known = sum(1 for leg in legs if leg.team)
        inferred = sum(1 for leg in legs if leg.team_source == "structural")
        if known == 0:
            lines.append(
                f"Rosters: no club known for any of the {len(legs)} ticket legs, "
                "so every same-game pair used the blended priors. "
                "`betedge parlay rosters` says why."
            )
        else:
            detail = f"Rosters: {known}/{len(legs)} ticket legs have a known club"
            if inferred:
                detail += f" ({inferred} inferred from the slate)"
            if book is not None and book.rejected_stale:
                detail += f"; {book.rejected_stale} entries dropped as stale"
            lines.append(detail + ".")
    return "\n".join(lines)


def sorted_by_ev(tickets: Sequence) -> list:
    """Expected value first. Never the payout multiple."""
    return sorted(tickets, key=lambda t: t.ev, reverse=True)


def sorted_by_ev_per_variance(tickets: Sequence) -> list:
    return sorted(tickets, key=lambda t: t.ev_per_variance, reverse=True)


def parlay_summary(result) -> str:
    took = (result.finished_at - result.started_at).total_seconds()
    lines = [
        f"Scanned {result.events_scanned} events across "
        f"{', '.join(result.sports) or 'nothing'} in {took:.0f}s",
        f"{result.legs_built:,} legs built, {result.legs_after_filter:,} survived "
        f"the filters, {result.groups_searched} group(s) searched",
        f"{result.candidates_evaluated:,} distinct tickets evaluated, "
        f"{len(result.clean)} clean, {len(result.suspect)} suspect",
        f"Products: {', '.join(result.products)}",
        f"Credits: {result.credits_spent} spent this run"
        + (
            f", {result.credits_remaining:,} remaining"
            if result.credits_remaining is not None
            else ""
        ),
    ]
    if result.legs_by_book:
        lines.append(
            "Legs by book: "
            + ", ".join(
                f"{book} ({n:,})"
                for book, n in sorted(
                    result.legs_by_book.items(), key=lambda kv: -kv[1]
                )
            )
        )
    for book in result.silent_books:
        # Named, not left as an absence. An empty board from a book you
        # configured means one of two completely different things -- no
        # usable lines today, or a book your feed does not carry at all --
        # and only one of them is worth waiting out.
        lines.append(
            f"No legs at all from {book}: it quoted nothing this scan could "
            f"use. `betedge parlay coverage` says whether your feed carries "
            f"it."
        )
    if result.rejections:
        top = sorted(result.rejections.items(), key=lambda kv: -kv[1])[:6]
        lines.append("Filtered out: " + ", ".join(f"{k} ({v:,})" for k, v in top))
    if result.errors:
        lines.append(f"{len(result.errors)} error(s): " + " | ".join(result.errors[:3]))
    return "\n".join(lines)


def _correlation_rows(ticket) -> str:
    rows = []
    for pair in ticket.correlation.pairs:
        a, b = ticket.legs[pair.i], ticket.legs[pair.j]
        badge = {
            "empirical": '<span class="src-measured">measured</span>',
            "prior": '<span class="src-prior">prior</span>',
            "default": '<span class="src-default">default</span>',
        }.get(pair.source, _esc(pair.source))
        sample = f"n={pair.sample_size:,}" if pair.sample_size else "&mdash;"
        rows.append(
            f"<tr><td class='dim'>{_esc(a.description)}</td>"
            f"<td class='dim'>{_esc(b.description)}</td>"
            f"<td class='dim'>{_esc(pair.relation.replace('_', ' '))}</td>"
            f"<td class='num'>{pair.rho:+.2f}</td>"
            f"<td>{badge}</td><td class='num dim'>{sample}</td></tr>"
        )
    if not rows:
        return ""
    return f"""<table class="inner">
<thead><tr><th>Leg</th><th>Leg</th><th>Relation</th><th class="num">&rho;</th>
<th>Source</th><th class="num">Sample</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""


def _leg_rows(ticket) -> str:
    rows = []
    for leg in ticket.legs:
        push = f"{leg.push_prob:.1%}" if leg.push_prob else "&mdash;"
        line_note = (
            '<span class="flag">interpolated</span>'
            if leg.line_source != "exact"
            else ""
        )
        if leg.team:
            team = (
                f"{_esc(leg.team)} "
                f"<span class='dim'>{_esc(leg.team_source or '')}</span>"
            )
        else:
            team = '<span class="dim">unknown</span>'
        rows.append(
            f"<tr><td class='bet'>{_esc(leg.description)}</td>"
            f"<td class='dim'>{_esc(pretty_market(leg.market))}</td>"
            f"<td class='dim'>{_esc(leg.matchup)}</td>"
            f"<td>{team}</td>"
            f"<td class='num'>{leg.fair_prob:.1%}</td>"
            f"<td class='num'>{push}</td>"
            f"<td class='num'>{leg.hit_prob:.1%}</td>"
            f"<td class='num dim'>{leg.sharp_price_taken:.2f} / "
            f"{leg.sharp_price_other:.2f}</td>"
            f"<td class='dim'>{_esc(leg.book)} {line_note}</td></tr>"
        )
    return f"""<table class="inner">
<thead><tr><th>Leg</th><th>Market</th><th>Game</th><th>Club</th>
<th class="num">Pinnacle fair</th><th class="num">Push</th>
<th class="num">Hits</th><th class="num">Pinnacle o/u</th><th>Book</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""


def _ticket_block(ticket, index: int) -> str:
    flags = "".join(f'<span class="flag">{_esc(f)}</span>' for f in ticket.flags)
    ident = ticket.db_id if getattr(ticket, "db_id", None) else index

    if ticket.correlation_is_load_bearing:
        verdict = (
            '<div class="alarm">This ticket is positive ONLY because of the '
            'assumed correlation. With the legs treated as independent it is '
            f'{ticket.ev_independent:+.1%}. The case for betting it is a '
            'correlation estimate, not a price.</div>'
        )
    elif ticket.correlation.all_prior:
        verdict = (
            '<div class="caution">Every correlation here comes from a '
            'structural prior, not from measured data. Treat the number as a '
            'hypothesis until the pairs have game logs behind them.</div>'
        )
    else:
        verdict = ""

    dist = "".join(
        f"<td class='num'>{p:.1%}</td>" for p in ticket.hit_distribution
    )
    dist_head = "".join(
        f"<th class='num'>{k}</th>" for k in range(ticket.n_legs + 1)
    )
    payout_row = "".join(
        f"<td class='num dim'>"
        f"{ticket.product.multiple_grid(ticket.n_legs)[0, k]:g}x</td>"
        for k in range(ticket.n_legs + 1)
    )

    return f"""<div class="ticket">
<div class="thead">
  <div><span class="tid">#{_esc(ident)}</span>
       <span class="bet">{ticket.n_legs}-leg {_esc(ticket.product.title)}</span>
       <span class="dim">{_esc(ticket.legs[0].matchup)}</span></div>
  <div><span class="ev">{ticket.ev:+.1%}</span>
       <span class="vs">vs {ticket.ev_independent:+.1%} independent</span></div>
</div>
{verdict}
<div class="stats">
  <span><b>P(all hit)</b> {ticket.joint_prob:.2%} &plusmn; {ticket.joint_prob_se:.2%}</span>
  <span><b>independence assumes</b> {ticket.joint_prob_independent:.2%}</span>
  <span><b>pays</b> {ticket.payout_all_hit:g}x</span>
  <span><b>EV/variance</b> {ticket.ev_per_variance:.3f}</span>
  <span><b>stake</b> {ticket.recommended_stake:,.0f}</span>
</div>
{_leg_rows(ticket)}
<div class="sub2">Correlation used, pair by pair</div>
{_correlation_rows(ticket)}
<div class="sub2">Payout structure &mdash; probability of exactly k legs hitting,
and what k pays</div>
<table class="inner"><thead><tr><th>legs hit</th>{dist_head}</tr></thead>
<tbody><tr><td class="dim">probability</td>{dist}</tr>
<tr><td class="dim">pays</td>{payout_row}</tr></tbody></table>
<div class="flags">{flags}</div>
</div>"""


_PARLAY_CSS = """
.ticket{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:14px 16px;margin-bottom:14px}
.thead{display:flex;flex-wrap:wrap;gap:10px;justify-content:space-between;
align-items:baseline;margin-bottom:8px}
.tid{color:var(--muted);font-variant-numeric:tabular-nums;margin-right:8px}
.vs{color:var(--muted);font-size:12.5px;margin-left:8px}
.stats{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--muted);
margin-bottom:12px}
.stats b{color:var(--ink);font-weight:600}
.alarm{background:var(--warn-bg);color:var(--warn);border-radius:8px;padding:9px 12px;
font-size:13px;margin-bottom:10px;line-height:1.5}
.caution{color:var(--muted);font-size:12.5px;margin-bottom:10px;line-height:1.5}
table.inner{border-collapse:collapse;width:100%;font-size:12.5px;margin-bottom:10px}
table.inner th{text-align:left;font-weight:600;color:var(--muted);font-size:10.5px;
text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;
border-bottom:1px solid var(--line);white-space:nowrap}
table.inner td{padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
.sub2{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em;
margin:12px 0 6px}
.src-measured{background:var(--pos-bg);color:var(--pos);border-radius:4px;
padding:1px 6px;font-size:11px}
.src-prior{background:var(--warn-bg);color:var(--warn);border-radius:4px;
padding:1px 6px;font-size:11px}
.src-default{color:var(--muted);font-size:11px}
.flags{margin-top:4px}
.tablewrap{overflow-x:auto}
"""


def parlay_report_html(result, cfg: Config, title: str = "Parlay scan") -> str:
    """
    The full ticket report.

    Ordered by expected value, never by payout multiple. A 20x ticket at
    -8% is a worse bet than a 3x at +4%, and a report sorted by multiple
    would say the opposite at a glance, which is the single easiest way to
    make this tool harmful.
    """
    stamp = result.finished_at.astimezone().strftime("%a %d %b %Y, %H:%M %Z")
    by_ev = sorted(result.clean, key=lambda t: t.ev, reverse=True)
    by_risk = sorted(result.clean, key=lambda t: t.ev_per_variance, reverse=True)

    total_stake = sum(t.recommended_stake for t in result.clean)
    cards = [
        ("Tickets", str(len(result.clean))),
        ("Suspect", str(len(result.suspect))),
        ("Best EV", f"{by_ev[0].ev:+.1%}" if by_ev else "—"),
        ("Legs built", f"{result.legs_built:,}"),
        ("Evaluated", f"{result.candidates_evaluated:,}"),
        ("Credits used", f"{result.credits_spent:,}"),
        ("Total stake", f"{total_stake:,.0f}"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v">{_esc(v)}</div></div>'
        for k, v in cards
    )

    def section(tickets, empty: str) -> str:
        if not tickets:
            return f'<div class="scroll"><div class="empty">{_esc(empty)}</div></div>'
        return "".join(_ticket_block(t, i) for i, t in enumerate(tickets, 1))

    risk_note = ""
    if by_risk and by_ev and by_risk[0] is not by_ev[0]:
        risk_note = (
            '<p class="note">The best ticket by expected value and the best by '
            'expected value per unit of variance are not the same ticket. They '
            'answer different questions &mdash; how much this makes, and how '
            'much it makes for the risk it carries &mdash; so both orderings '
            'are shown rather than blended into one score that answers '
            'neither.</p>'
        )

    rejected = ""
    if result.rejections:
        items = sorted(result.rejections.items(), key=lambda kv: -kv[1])
        rejected = "<br>".join(
            f"{_esc(k.replace('_', ' '))}: {v:,}" for k, v in items
        )
        rejected = (
            f'<h2>Why legs were filtered out</h2>'
            f'<p class="note">{rejected}</p>'
        )

    return f"""<title>Parlay Scan {stamp}</title>
<style>{_CSS}{_PARLAY_CSS}</style>
<div class="wrap">
<h1>{_esc(title)}</h1>
<p class="sub">{_esc(stamp)} &middot; {_esc(', '.join(result.sports))} &middot;
{_esc(', '.join(result.products))} &middot; marginals de-vigged from
{_esc(cfg.books.sharp)} ({_esc(cfg.model.devig_method)}) &middot;
{cfg.parlay.draws:,} Monte Carlo draws</p>
<div class="cards">{card_html}</div>

<h2>Ranked by expected value</h2>
{section(by_ev[: cfg.parlay.top_n], "Nothing cleared the thresholds. That is the normal result.")}

<h2>Ranked by expected value per unit of variance</h2>
{risk_note}
{section(by_risk[: cfg.parlay.top_n], "Nothing to rank.")}

<h2>Suspect</h2>
{section(result.suspect[:10], "Nothing flagged as suspect.")}
<p class="note">These tripped a guard &mdash; an edge too large to believe, a
Monte Carlo error comparable to the edge itself, an interpolated line, or an
expected value that only exists because correlation was assumed. No stake is
recommended for any of them.</p>

<p class="note"><b>Read the two EV numbers together.</b> The correlated figure is
what this model believes; the independent one is what the payout structure
assumes. The gap between them is the entire claim being made, and it rests on
correlation inputs that are marked, pair by pair, as measured or assumed.
Payout multiples come from
<code>{_esc(str(cfg.parlay.payouts_path or 'betedge/data/payouts.yaml'))}</code>
and are configuration, not fact &mdash; verify them against your account with
<code>betedge parlay verify-payouts</code> before acting on any number here.</p>
{rejected}
</div>"""
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

    # ---- systematic side lean -------------------------------------------
    sides = {"Over": 0, "Under": 0}
    for a in assessments:
        if a.ev <= 0:
            continue
        for name in sides:
            if a.description.endswith(name) or f" {name} " in a.description:
                sides[name] += 1
    if sum(sides.values()) >= 4:
        out.append("SIDE LEAN AMONG POSITIVE-EV QUOTES")
        out.append(f"  Over {sides['Over']}   Under {sides['Under']}")
        out.append("  A heavy lean is not noise. Soft books shade the side the "
                   "public bets,")
        out.append("  which on player props is the Over, leaving the Under "
                   "relatively better priced.")
        out.append("")

    # ---- the near misses ------------------------------------------------
    out.append(f"CLOSEST {top} TO CLEARING, BEST FIRST")
    header = (f"  {'':<6}{'EV':>8} {'bar':>7} {'short':>8}  {'liq':>5} {'orr':>6}  "
              f"{'bet':<44} {'price':>6}  game")
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    for a in sorted(assessments, key=lambda x: x.shortfall)[:top]:
        # Mark the ones that actually cleared. A negative shortfall means
        # cleared, which is correct and completely unreadable -- the whole
        # table looks like a list of bets when only the marked rows are.
        mark = "FLAG " if a.ev >= a.required_ev else "     "
        out.append(
            f"  {mark:<6}{a.ev:>+7.2%} {a.required_ev:>+6.2%} {a.shortfall:>+7.2%}  "
            f"{a.liquidity:>5.2f} {a.overround:>5.2%}  "
            f"{a.description:<44.44} {american(a.soft_price):>6}  {a.matchup[:28]}"
        )
    out.append("")
    n_flag = sum(1 for a in assessments if a.ev >= a.required_ev)
    if n_flag:
        out.append(f"Only the {n_flag} FLAG row(s) cleared. Everything else is "
                   f"listed by how close it came, not as a recommendation.")
    else:
        out.append("Nothing cleared. These are the closest misses, not bets.")
    return "\n".join(out)
