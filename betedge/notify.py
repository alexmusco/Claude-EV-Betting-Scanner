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
    get = row.get if isinstance(row, dict) else (lambda k, d=None: getattr(row, k, d))
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
        "Title": message.title,
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


def format_opportunity(row, stake=None, american=None) -> Message:
    """
    One bet, phrased to be actionable from a lock screen.

    Everything needed to place it and nothing else: what, where, at what
    price, how much. The reasoning belongs in the report -- a phone
    notification that has to be scrolled is one that gets dismissed.
    """
    get = row.get if isinstance(row, dict) else (lambda k, d=None: getattr(row, k, d))
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
    body_lines = [
        f"{get('book') or '?'} {shown}",
        str(get("matchup") or get("market") or ""),
    ]
    if stake:
        body_lines.insert(0, f"Stake {stake:,.0f}")
    commence = get("commence_time")
    if commence:
        body_lines.append(f"starts {commence}")
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


def format_digest(count: int, best_ev, spent=None) -> Message:
    """A single message standing in for several bets."""
    title = f"{count} bet(s) to place"
    if best_ev is not None:
        title += f", best {best_ev:+.1%}"
    body = "Run `bet show` for the list."
    if spent is not None:
        body += f"\n({spent} credits)"
    return Message(title=title, body=body, tags=["money_with_wings"])
