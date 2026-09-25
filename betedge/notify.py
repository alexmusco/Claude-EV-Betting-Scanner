"""
Getting a bet onto your phone.

The problem this solves
-----------------------
A scan that prints to a terminal is a scan you have to be sitting at. The
edges this tool finds decay -- a soft line that is three points off
Pinnacle at noon is usually gone by evening -- so the difference between
seeing it now and seeing it tonight is the difference between a bet and a
record of a bet you could have made.

Repeating yourself is worse than saying nothing
------------------------------------------------
Something running every fifteen minutes will find the SAME opportunity
every fifteen minutes. Twelve buzzes for one bet trains you to ignore the
thirteenth, which is the one that mattered. So every notification is
fingerprinted and stored, and the same bet is not sent twice unless
something about it actually changed:

  * the price improved by a material amount -- a better number on a bet
    you already took is worth knowing about
  * enough time passed that you may have missed the first one

That check lives in the database rather than in memory, because the whole
point is that this runs as a scheduled job: a fresh process every time,
with no recollection of what the last one said.

A notification failure is not a scan failure
--------------------------------------------
The scan is the valuable part and it has already cost API credits by the
time this runs. So every send is wrapped: a dead phone, a wrong token or
an unreachable host gets reported and recorded, and the caller keeps its
results.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone

#: Providers, in the order they are tried when none is named.
NTFY = "ntfy"
PUSHOVER = "pushover"
TELEGRAM = "telegram"
PROVIDERS = (NTFY, PUSHOVER, TELEGRAM)


class NotifyError(RuntimeError):
    """A notification could not be delivered."""


@dataclass
class Message:
    """One notification, in the shape a phone should show it."""

    title: str
    body: str
    #: Higher means it may break through a quiet setting on the phone.
    priority: int = 3
    tags: list[str] = field(default_factory=list)
    url: str | None = None

    def as_text(self) -> str:
        return f"{self.title}\n{self.body}"


def field(row, key, default=None):
    """
    One field out of a row, whatever kind of row it is.

    This exists because of a bug that survived two rounds of "fixing the
    notifications" and made every message useless:

        get = row.get if isinstance(row, dict) else lambda k, d: getattr(row, k, d)

    A `sqlite3.Row` is NOT a dict and does NOT support attribute access.
    It answers `row["selection"]` and `row.keys()` and nothing else. So
    that second branch returned the default for EVERY field of every row
    the database produced -- which is every row this module is ever
    called with in production. The notification body came out as "Stake
    5" because the stake was the only value not read off the row.

    Worse, the same accessor builds the dedupe fingerprint. Fed all
    Nones, every bet hashed identically, so after the first notification
    ever sent, every later bet looked like a repeat and was suppressed.

    Tests missed it for the oldest reason there is: they passed dicts,
    which take the branch that works.

    `keys()` is the test rather than the type, because it is what both
    dict and sqlite3.Row answer, and it keeps working for anything
    mapping-like added later. Objects with attributes still work.
    """
    if hasattr(row, "keys"):
        try:
            return row[key] if key in row.keys() else default
        except (KeyError, IndexError):
            return default
    return getattr(row, key, default)


def fingerprint(*parts) -> str:
    """
    A stable identity for "this bet", independent of price.

    Price is deliberately NOT in it. A line moving from -110 to -108 is
    the same bet, and fingerprinting the price would make every tick a
    new notification -- which is the failure this whole module exists to
    avoid. Whether a price move is worth re-sending is decided
    separately, by how far it moved.
    """
    blob = "|".join("" if p is None else str(p).strip().lower() for p in parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def opportunity_fingerprint(row) -> str:
    """The identity of a single-bet opportunity."""
    get = lambda k, d=None: field(row, k, d)  # noqa: E731
    return fingerprint(
        get("sport"), get("event_id"), get("market"),
        get("selection"), get("side"), get("line"), get("book"),
    )


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------


def _parse_clock(value) -> time | None:
    if value in (None, ""):
        return None
    if isinstance(value, time):
        return value
    text = str(value).strip()
    try:
        hour, _, minute = text.partition(":")
        return time(int(hour), int(minute or 0))
    except ValueError:
        return None


def in_quiet_hours(now: datetime, start, end) -> bool:
    """
    Whether `now` falls inside a quiet window, which may wrap midnight.

    Wrapping is the normal case -- nobody wants to be woken between 23:00
    and 08:00 -- and handling it as a simple `start <= t < end` would make
    the common configuration silently do nothing.
    """
    start_t, end_t = _parse_clock(start), _parse_clock(end)
    if start_t is None or end_t is None or start_t == end_t:
        return False
    current = now.time()
    if start_t < end_t:
        return start_t <= current < end_t
    return current >= start_t or current < end_t


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _post(url: str, session=None, timeout: float = 15.0, **kwargs):
    if session is not None:
        response = session.post(url, timeout=timeout, **kwargs)
    else:
        import requests

        response = requests.post(url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response


def _header_safe(text: str) -> str:
    """
    A string that will survive being sent as an HTTP header.

    Headers are latin-1, so a single accented letter in a player's name
    -- Amon-Ra St. Brown is fine, but the league is full of names that
    are not -- raises UnicodeEncodeError inside the HTTP client and
    takes the whole notification with it. The bet is worth more than the
    accent, so the accent goes.

    Newlines are stripped too: a header cannot hold one, and a library
    that does not notice would be splicing attacker-controlled-looking
    text into the request.
    """
    flat = " ".join(str(text or "").split())
    try:
        flat.encode("latin-1")
        return flat
    except UnicodeEncodeError:
        import unicodedata

        folded = unicodedata.normalize("NFKD", flat)
        stripped = "".join(c for c in folded if not unicodedata.combining(c))
        return stripped.encode("latin-1", "replace").decode("latin-1")


def send_ntfy(message: Message, topic: str, server: str = "https://ntfy.sh",
              session=None) -> None:
    """
    ntfy: no account, no key. The topic name IS the secret.

    Which is worth saying plainly: anyone who learns your topic can read
    your notifications and send you their own. Use a long random one, and
    do not put anything in a message you would mind a stranger reading.
    """
    if not topic:
        raise NotifyError("ntfy needs a topic")
    headers = {
        # `X-Title` is ntfy's canonical spelling; `Title` is an alias.
        # Prefer the canonical one -- a bare `Title` is the sort of
        # generic header an intermediary feels free to touch, and the
        # title is where the bet's name lives.
        "X-Title": _header_safe(message.title),
        "Priority": str(message.priority),
    }
    if message.tags:
        headers["Tags"] = ",".join(message.tags)
    if message.url:
        headers["Click"] = message.url
    _post(f"{server.rstrip('/')}/{topic}", session=session,
          data=message.body.encode("utf-8"), headers=headers)


def send_pushover(message: Message, token: str, user: str,
                  session=None) -> None:
    if not token or not user:
        raise NotifyError("pushover needs both a token and a user key")
    payload = {
        "token": token, "user": user,
        "title": message.title, "message": message.body,
        "priority": max(-2, min(1, message.priority - 3)),
    }
    if message.url:
        payload["url"] = message.url
    _post("https://api.pushover.net/1/messages.json", session=session,
          data=payload)


def send_telegram(message: Message, token: str, chat_id: str,
                  session=None) -> None:
    if not token or not chat_id:
        raise NotifyError("telegram needs both a bot token and a chat id")
    _post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        session=session,
        data={"chat_id": chat_id, "text": message.as_text(),
              "disable_web_page_preview": "true"},
    )


def _secret(cfg, name: str, env: str) -> str:
    """
    A credential from the environment first, then the config file.

    Environment first because a config file gets committed by accident
    and an environment variable does not.
    """
    return str(os.environ.get(env) or getattr(cfg, name, "") or "").strip()


def send(message: Message, cfg, session=None) -> str:
    """
    Deliver one message, returning the provider that took it.

    Raises NotifyError rather than returning a failure code, because a
    caller that ignores the return value would silently stop notifying
    and look exactly like a quiet day.
    """
    provider = (getattr(cfg, "provider", "") or "").strip().lower()
    if not provider:
        raise NotifyError(
            "no notification provider configured. Set notify.provider to "
            f"one of {', '.join(PROVIDERS)}."
        )
    if provider == NTFY:
        send_ntfy(message, _secret(cfg, "ntfy_topic", "BETEDGE_NTFY_TOPIC"),
                  server=getattr(cfg, "ntfy_server", "https://ntfy.sh"),
                  session=session)
    elif provider == PUSHOVER:
        send_pushover(
            message,
            _secret(cfg, "pushover_token", "BETEDGE_PUSHOVER_TOKEN"),
            _secret(cfg, "pushover_user", "BETEDGE_PUSHOVER_USER"),
            session=session,
        )
    elif provider == TELEGRAM:
        send_telegram(
            message,
            _secret(cfg, "telegram_token", "BETEDGE_TELEGRAM_TOKEN"),
            _secret(cfg, "telegram_chat_id", "BETEDGE_TELEGRAM_CHAT_ID"),
            session=session,
        )
    else:
        raise NotifyError(
            f"unknown notification provider {provider!r}. "
            f"Expected one of {', '.join(PROVIDERS)}."
        )
    return provider


# ---------------------------------------------------------------------------
# What to say
# ---------------------------------------------------------------------------


def _when(value) -> str:
    """
    A start time a person can read at a glance, in THEIR timezone.

    "starts 2026-09-18T00:15:00Z" tells a phone user nothing they can
    act on -- it is a UTC timestamp for a decision measured in how many
    minutes are left. Anything unparseable falls back to the raw value
    rather than vanishing.
    """
    from datetime import datetime, timezone

    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone()
    minutes = (local - datetime.now(local.tzinfo)).total_seconds() / 60.0
    stamp = local.strftime("%a %-I:%M %p") if hasattr(local, "strftime") else str(local)
    if 0 < minutes < 600:
        return f"{stamp} (in {minutes/60:.1f}h)" if minutes >= 60 \
            else f"{stamp} (in {minutes:.0f}m)"
    return stamp


def format_opportunity(row, stake=None, american=None) -> Message:
    """
    One bet, phrased to be actionable from a lock screen.

    Everything needed to place it and nothing else: what, where, at what
    price, how much. The reasoning belongs in the report -- a phone
    notification that has to be scrolled is one that gets dismissed.
    """
    get = lambda k, d=None: field(row, k, d)  # noqa: E731
    bits = [
        str(get("selection") or "").strip(),
        str(get("side") or "").strip(),
        "" if get("line") is None else f"{float(get('line')):g}",
    ]
    what = " ".join(b for b in bits if b)
    price = get("soft_price")
    shown = american(price) if (american and price) else price
    ev = get("ev")

    title = f"{ev:+.1%}  {what}" if ev is not None else what

    # THE BET GOES FIRST, IN THE BODY.
    #
    # Not in the title alone. For ntfy the title travels as an HTTP
    # HEADER, and a header can be dropped, truncated or mangled by any
    # hop between here and the phone -- which is exactly what happened:
    # a notification arrived reading only "Stake 5", because the stake
    # was the first line of the body and the bet itself existed nowhere
    # else. The body is the POST payload: UTF-8, unambiguous, and it
    # always arrives.
    #
    # So the body opens with what to place, then how much and at what
    # price. A lock screen that collapses to one line still shows the
    # only line that matters.
    first = what or str(get("market") or "a bet")
    if ev is not None:
        first = f"{first}   ({ev:+.1%})"
    body_lines = [first]

    money = f"{get('book') or '?'} {shown}"
    if stake:
        money += f"   stake {stake:,.0f}"
    body_lines.append(money)

    body_lines.append(str(get("matchup") or get("market") or ""))
    commence = get("commence_time")
    if commence:
        body_lines.append(f"starts {_when(commence)}")
    opportunity_id = get("id")
    if opportunity_id:
        # So it can be logged without hunting for the id afterwards.
        body_lines.append(f"bet bet {opportunity_id} --stake {stake or 0:,.0f}")

    return Message(
        title=title,
        body="\n".join(line for line in body_lines if line),
        priority=4 if (ev or 0) >= 0.05 else 3,
        tags=["money_with_wings"],
    )


def digest_line(row, stake=None, american=None) -> str:
    """One bet, compressed to a single readable line for a digest."""
    get = lambda k, d=None: field(row, k, d)  # noqa: E731
    bits = [
        str(get("selection") or "").strip(),
        str(get("side") or "").strip(),
        "" if get("line") is None else f"{float(get('line')):g}",
    ]
    what = " ".join(b for b in bits if b) or str(get("market") or "a bet")
    price = get("soft_price")
    shown = american(price) if (american and price) else price
    ev = get("ev")
    line = f"{ev:+.1%}  {what}" if ev is not None else what
    tail = " ".join(str(x) for x in [get("book") or "", shown or ""] if x)
    if tail:
        line += f"  ({tail}"
        line += f", {stake:,.0f})" if stake else ")"
    return line


def format_digest(count: int, best_ev, spent=None, rows=None,
                  american=None) -> Message:
    """
    A single message standing in for several bets -- WITH the bets in it.

    This used to read "Run `bet show` for the list", which sends you to a
    terminal to find out what your phone already knew. The whole point of
    notifying is to be actionable from a lock screen anywhere; a message
    whose content is an instruction to go and look somewhere else is a
    message that may as well not have been sent.

    So the digest lists them. It exists to stop a phone buzzing nine
    times, not to withhold nine bets.
    """
    title = f"{count} bet(s) to place"
    if best_ev is not None:
        title += f", best {best_ev:+.1%}"

    lines = []
    for entry in (rows or []):
        row, stake = entry if isinstance(entry, tuple) else (entry, None)
        lines.append(digest_line(row, stake=stake, american=american))
    if not lines:
        lines = [f"{count} bet(s) cleared the bar. Run `bet show` for the list."]
    if spent is not None:
        lines.append(f"({spent} credits)")
    return Message(title=title, body="\n".join(lines),
                   tags=["money_with_wings"])
