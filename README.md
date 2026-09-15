# betedge

Finds positive expected value bets by stripping the vig out of a sharp
book's prices (Pinnacle) and comparing the result to softer books
(DraftKings, Underdog). Logs everything it flags, tracks what you actually
bet, and measures closing-line value so you can tell whether the model
works long before the profit and loss says so.

Rebuilt from the R prop scanner, with the pricing fixed and guards added.

---

## Setup

```bash
bash setup.sh
```

That's it. The script builds the virtual environment, installs
dependencies, checks the API key in `.env`, finds your database if it got
left behind in an older copy of the folder, and offers to add the `bet`
shell alias with the right absolute path filled in.

It `cd`s to its own directory first, so it works no matter which folder you
run it from — which is the usual cause of
`no such file or directory: .venv/bin/activate`.

**Run it again after every update.** New versions sometimes add a
dependency, and this is what puts it in place.

Doing it by hand instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m betedge quota          # free — confirms the key works
```

---

## Daily use

```bash
python -m betedge scan                     # scan the configured sports
python -m betedge scan --sports basketball_nba --min-ev 0.03
python -m betedge scan --max-events 5      # cap the credit burn

python -m betedge bet 42 --stake 150       # log a bet on opportunity #42
python -m betedge bet 42 --stake 150 --price 2.05   # if the price moved

python -m betedge settle 7 won             # won | lost | push | void | half_won | half_lost

python -m betedge close                    # capture closing lines
python -m betedge report                   # performance and CLV
python -m betedge export bets.csv
```

Every scan writes `reports/scan_<timestamp>.html` and records itself in
`data/betedge.db`.

Run `close` on a schedule so closing lines actually get captured — props
are usually pulled at tip-off, so there is no catching up afterwards:

```
*/15 * * * * cd /path/to/betedge && .venv/bin/python -m betedge close
```

---

## What changed from the R version

**The pricing was wrong.** The old model compared raw implied
probabilities between books:

```r
over_diff <- (1/pinnacle_over_price) - (1/underdog_over_price)
```

`1/price` includes the book's margin, so this is a comparison of two
vigged numbers. It tells you the books disagree; it does not tell you by
how much, or in whose favour. The threshold — half the slate's average
Pinnacle vig — then moved around with whatever margin happened to be on
the board that night.

The fix is to de-vig Pinnacle first, recovering a fair probability `p`,
then compute the actual expected value of the soft book's price `d`:

```
EV = p × d − 1
```

That number means something: `+0.04` is four cents of edge per dollar
staked. You can threshold it, rank on it, size bets from it, and check it
against closing lines later.

**Four de-vig methods, and the choice matters more than you'd think.**
Multiplicative, additive, power, and Shin are all implemented. On a
near-even prop they agree to within half a percentage point. On a lopsided
one they do not: a 1.05 / 9.50 market puts the longshot at 9.9% under the
multiplicative method and 5.9% under the power method. That difference is
the entire edge on most alternate lines and anytime-scorer props. The
default is `worst_case` — take the lowest fair probability any method
gives, so a bet only clears the bar if it clears it on every method.

(Worth knowing: for two-outcome markets, Shin is algebraically identical to
the additive method. The tests verify this to 1e-13. It only earns its keep
on three-way soccer markets.)

**Sanity guards.** The old model had one good guard —
`UD_line == PIN_line`, which is kept. Added:

| Guard | Why |
|---|---|
| EV above 15% → suspect | Real prop edges are 1–5%. A 20% edge is a stale line, a pulled market, or the wrong player. |
| Soft or sharp quote older than 20 min → suspect | You are pricing against a number that no longer exists. |
| Pinnacle overround outside 0.5–15% → skip | Malformed or closing market. |
| De-vig methods disagree by >4pp → skip | Your fair estimate is method-dependent, so it isn't an estimate. |
| Edge positive under some methods, negative under others → suspect | Not a real edge. |
| Event starts in under 5 min → skip | You will not get it down. |

Suspect rows are still shown — they are often the interesting ones — but
get no stake recommendation.

**Staking.** The old model had none. Now: quarter Kelly, capped at 2% of
bankroll per bet, with a 15% cap on total simultaneous exposure. Full Kelly
assumes you know `p` exactly; you don't, and overestimating the edge is how
Kelly bettors go broke. Props on one slate are also correlated, so sizing
each independently quietly overbets the portfolio — hence the total cap.

**Credit accounting.** The old scripts sent `regions=us,eu` *and*
`bookmakers=...`. Cost is markets × regions, and up to ten bookmakers count
as one region, so that doubled the bill of every call for nothing. Sending
only `bookmakers` halves the cost. The client also reads `x-requests-last`
off each response, so it knows the real spend rather than estimating, and
stops when a per-scan budget or a remaining-credit floor is hit.

**Tracking.** SQLite instead of the Excel sheet. Every flagged opportunity
is recorded, not just the bets you take — see below.

---

## Excel tracker export

```bash
python -m betedge export ~/Desktop/AlexBetTracker.xlsx
```

Any path ending `.xlsx` fills your existing tracker from the database
instead of writing a CSV. It writes only the input columns — date, sport,
bet type, wager, odds, win, pick, specifics, strategy — and leaves every
formula on all three sheets untouched, so yield, payoff ratio, Kelly,
Sharpe, t-stat, p-value and rolling P&L all still compute as you built
them. Rerun it any time; it always writes a fresh copy from the database.

Only **won** and **lost** bets are exported. The sheet's P&L formula is
`=IF(I=1, G*(H-1), -G)`, which treats a blank Win cell as a full loss, so
exporting a pending bet would book a phantom loss. Pushes, voids and
half-settled bets have no representation in that formula at all — they are
skipped and counted rather than fudged into a number that would corrupt
the t-stat.

**Two bugs fixed in the template while wiring this up**, both pre-existing:

- `# of Bets` was `=COUNT('1. Bet Entry'!J:J)`. Column J holds a formula in
  all 294 pre-filled rows and evaluates to 0 when the row is empty, so
  COUNT returned **294 no matter how many bets you had**. That fed average
  bet size, the t-stat, the p-value and "1 in X" — so with 7 real bets,
  average bet size read $0.26 instead of $10.71. Now counts column G.
- `Return` was `=J/G`, which is `#DIV/0!` on every empty row. That error
  propagated into Avg Return, STDEV and Sharpe, leaving all three broken.
  Now `=IFERROR(J/G,"")`.

Verified by recalculating the exported workbook in LibreOffice and reading
the analytics back.

---

## Why closing-line value is the number to watch

Profit and loss over 50 bets tells you almost nothing. At a 3% edge with
even-money props, several hundred bets are needed before the signal clears
the noise. You will have losing months while the model is working fine, and
winning months while it is broken.

Closing-line value cuts through that. Take a bet at 2.15, and by tip-off
the de-vigged sharp price is 2.05 — you got a better number than the market
settled on, and that is *observable immediately*, with no waiting for the
result. Consistently beating the close is the strongest available evidence
that a model is finding real mispricing. Consistently landing at zero CLV
while showing a profit means you have been lucky.

So `betedge` logs every opportunity it flags, not only the ones you bet,
and `betedge close` captures the sharp price at the end. `betedge report`
shows average CLV, the share of bets that beat the close, and your P&L
against what the model expected.

The honest framing: this strategy is well known, the edges are small, and
most of the work is execution — getting a bet down before the soft book
moves. Expect that DraftKings limits accounts that beat it consistently.
None of the guards here can protect you from betting more than you should.

---

## Two kinds of market, very different economics

**Player props** (`sports:` in the config) come off the per-event endpoint
and cost markets × events. An NFL Sunday is ~180 credits; a full MLB day
across ten markets is ~150.

**Game-level markets** (`core_sports:`) — moneyline, spreads, totals — come
off the bulk endpoint, where cost is markets × 1 for the **entire sport**.
Every tennis match on the board, all three markets, is 3 credits. Not 3 per
match. Three.

So mainlines are roughly fifty times cheaper per opportunity. The tradeoff
is real: edges are thinner there, because mainlines are where the sharp
money concentrates and the soft books pay most attention. Expect more +2.5%
plays and fewer +6% ones. But at 3 credits a sweep you can scan constantly,
and volume at a real 2.5% beats scarcity at an imagined 6%.

```bash
python -m betedge scan --core-sports "tennis_*" --no-props
python -m betedge scan                         # both passes, per config.yaml
```

A trailing `*` matches sport keys by prefix. This matters for tennis, which
publishes one key per tournament (`tennis_atp_china_open`,
`tennis_wta_wuhan`, …) and rotates them weekly — `tennis_*` is the only way
to say "all tennis" that does not need editing every month. Wildcards are
resolved against the live sport list, which is a free call.

### How one code path handles all of it

De-vigging needs a complete set of mutually exclusive outcomes. Those come
in three shapes and the engine keys them uniformly:

| Market | Outcomes named | Grouped by | Outcome identified by |
|---|---|---|---|
| Player prop | Over / Under | market, player, line | the side |
| Game total | Over / Under | market, line | the side |
| Moneyline | the competitors | market | competitor name |
| Spread / handicap | the competitors | market, \|handicap\| | competitor **and** handicap |

Two consequences worth knowing. A spread is only ever compared against the
same handicap — DraftKings on −2.5 is never matched to Pinnacle on −3.5,
the same guard that protects the props. And three-way markets need no
special case: soccer's 1X2 is just a group with three outcomes, and the
n-way de-vig handles it (this is where Shin stops being identical to the
additive method and starts earning its keep).

Completeness is checked by the overround guard rather than by counting
outcomes. An incomplete set — a one-sided prop, a 1X2 missing its draw —
sums to less than 1 before de-vigging, which falls outside the configured
overround band and is rejected. One rule, every market shape.

## Narrowing the prop markets

Cost is markets × events, so the market list is the single biggest lever on
what a scan costs. `prop_markets` in the config overrides the registry
default for one sport:

```yaml
prop_markets:
  baseball_mlb:
    - pitcher_strikeouts
    - batter_total_bases
```

A full MLB slate across all ten configured markets is ~150 credits a day.
Those two markets bring it to ~30. Sports left out of `prop_markets` use the
full list. `--markets pitcher_strikeouts batter_total_bases` does the same
thing for one run.

### MLB specifically

Lineups are the whole game. Batter props void if the player doesn't start,
and pitcher props void on a scratched starter — so before lineups post
(roughly 2–4 hours before first pitch) the numbers are placeholders and
Pinnacle's limits are low. **Scan after lineups, not before.** A morning
scan of an evening slate is mostly noise.

Run lines are almost always ±1.5, which makes MLB the sport where the
spread guard matters least and the totals guard matters most.

## Sports

Configured in `markets.py`; run `python -m betedge sports` for the live list.

| Sport | Props | Note |
|---|---|---|
| Tennis | none | **Core markets.** Pinnacle is *the* reference book for tennis and US books lag it badly. Two-way moneylines, so the existing math applies directly. Use `tennis_*`. |
| NBA | 12 markets | Deepest coverage both sides. Most picked-over, so edges are small but frequent. |
| NFL | 13 markets | Props post days early; soft books lag midweek injury and weather news. |
| NHL | 5 markets | Shots on goal and blocked shots are reliably soft. Lower volume, slower correction. |
| MLB | 10 markets, narrow it | Pitcher strikeouts and batter total bases are the targets; the other eight aren't worth the credits. Games every day, Apr–Oct. Core markets are 3 credits for the whole slate. |
| EPL / UCL / La Liga / Serie A / Bundesliga | 3 markets | Pinnacle is very sharp on soccer but its *prop* coverage is thin. Expect few matches. |
| NCAAB / NCAAF | inherited | The edge is in games nobody watches. Prop coverage patchy. |
| MMA | none | Moneyline only, but Pinnacle is sharp and soft books are slow on fight-week news. |

On EPL specifically: prop scanning will run but mostly return nothing,
because Pinnacle prices few player props for soccer. The genuine soccer
edge is in core markets — 1X2, totals, Asian handicaps — so put soccer in
`core_sports`, not `sports`.

---

## Layout

```
betedge/
  pricing.py    de-vig, EV, Kelly          ← the maths, heavily tested
  markets.py    sport and market registry, credit cost rules
  oddsapi.py    API client, quota tracking, caching, retries
  scan.py       parse → group → score → guard; props and game markets
  closing.py    closing-line capture
  db.py         SQLite store and analytics
  report.py     console tables and HTML reports
  cli.py        commands
tests/          287 tests; fixtures cover each decision boundary
```

```bash
pip install -r requirements-dev.txt
pytest
```
