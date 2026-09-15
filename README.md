# betedge

Finds positive expected value bets by stripping the vig out of Pinnacle's
prices and comparing the result to DraftKings. Ranks what it finds by how
much the number can be trusted, sizes the bets, paces itself against a
monthly credit budget, and measures closing-line value so you can tell
whether the model works long before the profit and loss says so.

---

## Setup

```bash
bash setup.sh
```

That's it. The script builds the virtual environment, installs
dependencies, checks the API key in `.env`, finds your database if it got
left behind in an older copy of the folder, and offers to add the `bet`
shell alias with the right absolute path filled in.

**Run it again after every update.** New versions sometimes add a
dependency, and this is what puts it in place.

Then set one thing — `bankroll.amount` in `config.yaml` — to what you are
actually willing to lose. Everything else has a defensible default.

---

## Daily use

One command:

```bash
bet daily
```

It works out what today's credits allow, captures closing lines for bets
you already have on, scans cheapest-markets-first inside that allowance,
and prints a shortlist:

```
Cycle       01 Sep to 01 Oct   (15.8 days left)
Remaining   18,000 of 20,000 (API)
Spent       2,000 this cycle, 0 today
Pace        1,091/day even; today's allowance 2,181
Available   2,181 credits for this run

 id      EV  liq    bet                              price   stake  game
------------------------------------------------------------------------
  1  +5.0%  deep   Kansas City Chiefs ML       2.10 (+110)      25  DEN @ KC (16.5h)
  2  +7.2%  thin   Y. Yamamoto Over 6.5        2.10 (+110)      35  SD @ LAD (12.5h)
  3  +3.0%  deep   Kansas City Chiefs ML       2.06 (+106)      15  DEN @ KC (16.5h)

3 playable, showing 3. Total recommended stake 75.
The id column is what `betedge bet <id> --stake <amount>` takes.
```

Then log what you actually got down:

```bash
bet bet 1 --stake 25              # the id from the list
bet bet 1 --stake 25 --price -110 # if the price moved before you clicked
bet settle 1 won                  # won | lost | push | void | half_won | half_lost
```

Prices are shown and accepted in American, because that is what
DraftKings puts on the screen. `--price` takes either format — anything
negative or at/beyond ±100 is read as American, anything between 1 and 100
as decimal, and the two ranges do not overlap for any realistic price.
Decimal is what gets stored, because that is what the maths uses.

And periodically:

```bash
bet budget          # credits left, today's allowance, recent spending
bet quota           # what your config costs, and how often you can run it
bet report          # performance and closing-line value
bet show            # re-print the last scan
bet export ~/Desktop/AlexBetTracker.xlsx
```

### Run it hourly, not daily

The name is misleading in one direction: `daily` is safe to run as often as
you like, because the budget governor decides what each run may spend. And
you should, because a soft-book edge often lives for minutes rather than
hours. A game-level sweep of the whole configured board costs about 19
credits, which on a 20,000 plan is roughly 33 sweeps an hour if you wanted
them.

```
0 * * * * cd /path/to/betedge && .venv/bin/python -m betedge daily --no-report >> data/cron.log 2>&1
```

`bet quota` tells you exactly what your configuration affords.

---

## The credit budget

The Odds API sells credits monthly. 20,000 sounds generous until one
careless prop scan of a full NFL Sunday across every market takes 2,400 of
them, or a cron job stuck in a loop takes the rest.

There were already two guards: a ceiling on what one scan may cost, and a
floor of credits it refuses to drop below. Both catch a single runaway
call. Neither catches the slower failure, which is spending three weeks of
quota in the first four days and going dark.

So spend is now paced against a billing cycle:

```
remaining credits ÷ days left  →  even pace
even pace × burst factor       →  today's allowance
allowance − spent today        →  what this run may cost
```

Three things worth knowing about how it behaves:

- **The provider's count is the truth.** Every API response carries
  `x-requests-remaining`. That is what `remaining` uses, so the plan cannot
  drift the way a local tally would. The local ledger (`credit_spend` in
  the database) records where credits went, which the header cannot tell
  you.

- **Unspent credits roll forward with no special handling.** A quiet day
  leaves `remaining` higher than the even pace assumed, so tomorrow's pace
  rises on its own.

- **Overspending throttles itself.** Burn two-thirds of the month in the
  first week and the remaining pace drops accordingly — no intervention,
  no separate alarm.

The burst factor (default 2.0) lets a day spend twice the even pace,
because opportunity is not spread evenly: an NFL Sunday is worth more
credits than a Tuesday in September. A reserve (default 800) is held back
so closing-line capture is never starved by scanning — CLV is the only
fast evidence the model works, so it gets paid first.

---

## Why the list is not sorted by EV

Ranking opportunities by raw expected value sorts the board in almost
exactly the wrong order.

The biggest apparent edges cluster in the thinnest markets — alternate
lines, obscure props, markets posted days before the event — because that
is where Pinnacle's own price is least certain and where a stale quote
survives longest. A scanner that ranks on EV alone hands you a list sorted
by *how likely this number is to be wrong*, which correlates with, but is
not, *how much money is here*.

The fix is to stop treating the de-vigged Pinnacle probability as truth and
start treating it as an estimate with error. That error is small on a
market Pinnacle will take $50,000 on and large on one it will take $250 on.
A +3% edge against a high-limit number is worth more than a +6% edge
against a low-limit one, because the 6% is mostly estimation error.

The API does not publish limits. Three observable proxies stand in:

| Signal | Why it works |
|---|---|
| **Pinnacle's own margin**, per outcome *and per tier* | Pinnacle sets margin inversely to limit as policy. But the comparison has to be against normal **for that kind of market**: it charges ~1.25% a side on a game line and ~3.5% on an MLB prop, so an absolute scale marks every prop as thin for a reason that is simply what props cost. What matters is whether this market is unusually wide *for its kind*. |
| **Market tier** | Game sides and totals are where sharp money concentrates. Primary props are heavily bet but an order of magnitude thinner. Alternate lines are thinner again. |
| **Time to start** | Limits rise and prices converge as an event approaches. A prop posted Tuesday for a Sunday game is a placeholder. |

A fourth adjustment is about the maths rather than the market: on lopsided
prices the de-vig methods disagree most, so a fair probability near 0 or 1
carries more model risk regardless of liquidity. Anytime-touchdown and
similar yes/no longshots live here.

These combine into a 0–1 score used two ways:

- **A sliding EV bar.** Game lines flag at **+2%**, primary props at about
  **+2.8%**, alternate lines higher still. This is the Bayesian-correct
  response to a noisier estimate, not timidity: if your fair probability
  could be off by two points, a two-point edge is not an edge.

- **Ranked ordering.** `edge_score = EV × liquidity` is what the reports
  sort on. A +3% NFL side scores 0.030 and outranks a +5% alternate prop at
  0.018.

Nothing is thrown away — raw EV is still recorded and still shown, and the
`liq` column tells you which kind of bet you are looking at. Set
`model.liquidity_ev_penalty: 0` for one flat bar across every market, or
`model.min_liquidity: 0.30` to drop the thin stuff entirely.

This is also, incidentally, the honest answer to "is more volume better?"
Yes — but not because thin markets are less profitable in principle. It is
because in a thin market you cannot tell profit from noise, and the number
you are betting into is one the sharp book itself is not confident in.

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
the additive method. The tests verify this to 1e-12. It only earns its keep
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
| EV below the liquidity-adjusted bar → skip | See above. |

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
only `bookmakers` halves the cost. The client reads `x-requests-last` off
each response, so it knows the real spend rather than estimating.

**Tracking.** SQLite instead of the Excel sheet. Every flagged opportunity
is recorded, not just the bets you take — see below.

---

## Two kinds of market, very different economics

**Game-level markets** (`core_sports:`) — moneyline, spreads, totals — come
off the bulk endpoint, where cost is markets × 1 for the **entire sport**.
Every tennis match on the board, all three markets, is 3 credits. Not 3 per
match. Three.

**Player props** (`sports:`) come off the per-event endpoint and cost
markets × events. A full MLB slate across two markets is ~30 credits;
across all ten it is ~150. An NFL Sunday across five markets is ~70.

So mainlines are roughly fifty times cheaper per opportunity. They are also
the *deeper* markets, which under the liquidity scoring means they clear
the bar at +2% rather than +3% or more. The old tradeoff framing — cheap
but thinner edges — was only half right: the edges are smaller, but they
are also far more likely to be real.

The scan runs the cheap pass first, deliberately. Whichever pass runs
second is the one a tight budget truncates, and it should not be the deep,
well-priced one.

```bash
bet daily                                      # both passes, per config.yaml
bet scan --core-sports "tennis_*" --no-props   # game lines only
bet scan --markets pitcher_strikeouts          # one prop market, one run
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
n-way de-vig handles it.

Completeness is checked by the overround guard rather than by counting
outcomes. An incomplete set — a one-sided prop, a 1X2 missing its draw —
sums to less than 1 before de-vigging, which falls outside the configured
overround band and is rejected. One rule, every market shape.

---

## Keeping the credit bill down

Three levers, in order of effect:

**1. The prop market list.** Cost is markets × events, so this is the
single biggest one. `prop_markets` overrides the registry default per
sport:

```yaml
prop_markets:
  baseball_mlb:
    - pitcher_strikeouts
    - batter_total_bases
```

Ten markets to two takes a full MLB slate from ~150 credits to ~30.

**2. The prop look-ahead window.** `prop_windows` caps how far ahead the
prop pass looks, per sport. This filter runs on the free event list, so
every event it drops is a per-event call never billed.

```yaml
prop_windows:
  baseball_mlb: 8
```

MLB is the case that matters. Batter props void if the player doesn't
start and pitcher props void on a scratched starter, so before lineups post
— roughly 2–4 hours before first pitch — the numbers are placeholders and
Pinnacle's limits are low. **Scan after lineups, not before.** A morning
scan of an evening slate is mostly noise you paid for.

**3. The core market list.** The bulk endpoint bills for markets
*requested*, not returned, so asking a sport for markets it doesn't price
is pure waste:

```yaml
core_markets:
  mma_mixed_martial_arts:
    - h2h
```

---

## Sports

Configured in `markets.py`; run `bet sports` for the live list.

| Sport | Props | Note |
|---|---|---|
| Tennis | none | **Core markets.** Pinnacle is *the* reference book for tennis and US books lag it badly. Two-way moneylines, so the existing math applies directly. Use `tennis_*`. |
| NBA | 12 markets | Deepest coverage both sides. Most picked-over, so edges are small but frequent. |
| NFL | 13 markets | Props post days early; soft books lag midweek injury and weather news. |
| NHL | 5 markets | Shots on goal and blocked shots are reliably soft. Lower volume, slower correction. |
| MLB | 10 markets, narrow it | Pitcher strikeouts and batter total bases are the targets. Games every day, Apr–Oct. Core markets are 3 credits for the whole slate. |
| EPL / UCL / La Liga / Serie A / Bundesliga | 3 markets | Pinnacle is very sharp on soccer but its *prop* coverage is thin. Put soccer in `core_sports`, not `sports`. |
| NCAAB / NCAAF | **blocked** | Oregon prohibits collegiate wagering, so these are excluded in the engine, not just left out of the config. See below. |
| MMA | none | Moneyline only, so `core_markets` narrows it to 1 credit. Pinnacle is sharp and soft books are slow on fight-week news. |

### Sports you cannot bet

Oregon permits no collegiate wagering, so DraftKings will not take an NCAA
bet. Scanning college is worse than useless — it spends credits surfacing
edges you cannot act on and pushes real bets down the list.

`excluded_sports` in `config.yaml` blocks them, and it is enforced in the
scan engine rather than by omission from the sport lists:

```yaml
excluded_sports:
  - "*ncaa*"
```

### Calibrating the liquidity model

`betedge diagnose` prints Pinnacle's overround quartiles by tier. Those are
what `TIER_OVERROUND_BASELINE` in `liquidity.py` should match. The shipped
values were measured off a live board in September 2026 — 15 MLB games, 236
DraftKings quotes:

| Tier | Total overround (25th / median / 75th) | Per outcome |
|---|---|---|
| mainline | 2.27% / 2.50% / 2.86% | ~1.25% |
| primary_prop | 6.92% / 7.03% / 7.16% | ~3.52% |

Re-measure when a season changes or a new sport comes on the board. If your
`diagnose` output disagrees materially with the table above, the baselines
are stale and every liquidity score built on them is off.

That placement is the point. Leaving college out of `core_sports` would be
undone by `--core-sports americanfootball_*`, or by `--sports
basketball_ncaab`, or by a wildcard resolving against a live sport list
that happens to include a college key. The check sits below all three, so
no billed call is ever made for a blocked sport. `bet quota` marks them
`excluded` rather than pricing them.

Patterns take a wildcard anywhere — `*ncaa*` is needed because
`americanfootball_ncaaf`, `basketball_ncaab`, `basketball_wncaab` and
`baseball_ncaa` share no usable prefix.

---

## Excel tracker export

```bash
bet export ~/Desktop/AlexBetTracker.xlsx
```

**One-time setup:** this fills a copy of *your* workbook, so it needs the
workbook to copy from. Put a blank copy at
`betedge/templates/tracker_template.xlsx`, or pass `--template
~/Desktop/blank_tracker.xlsx` per run. CSV export needs no template.

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
and captures the sharp price at the end. `bet report` shows average CLV,
the share of bets that beat the close, and your P&L against what the model
expected.

`bet daily` captures closing lines on every run, before it scans. If you
are running it hourly that is enough. If you run it rarely, add:

```
*/15 * * * * cd /path/to/betedge && .venv/bin/python -m betedge close
```

The honest framing: this strategy is well known, the edges are small, and
most of the work is execution — getting a bet down before the soft book
moves. Expect that DraftKings limits accounts that beat it consistently.
None of the guards here can protect you from betting more than you should.

---

## Layout

```
betedge/
  pricing.py    de-vig, EV, Kelly          ← the maths, heavily tested
  liquidity.py  how far to trust a fair price, and what edge to demand
  budget.py     monthly credit pacing
  markets.py    sport and market registry, credit cost rules
  oddsapi.py    API client, quota tracking, caching, retries
  scan.py       parse → group → score → guard; core markets then props
  closing.py    closing-line capture
  db.py         SQLite store, credit ledger, analytics
  report.py     console tables and HTML reports
  tracker.py    Excel tracker export
  cli.py        commands
tests/          175 tests; no network, no credits spent
```

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Commands

| Command | Cost | What it does |
|---|---|---|
| `bet daily` | budgeted | Closing lines, then a budgeted scan, then a shortlist. The one to run. |
| `bet budget` | free | Credits left, today's allowance, recent spending |
| `bet quota` | free | What your config costs and how often you can run it |
| `bet sports` | free | Live sport keys and prop coverage |
| `bet scan` | varies | A scan with manual flags, ignoring the daily plan |
| `bet show` | free | Re-print a scan |
| `bet bet <id> --stake N` | free | Log a bet you placed |
| `bet settle <id> won` | free | Settle it |
| `bet close` | ~1/event | Capture closing lines |
| `bet report` | free | Performance and CLV |
| `bet export <path>` | free | CSV, or your Excel tracker if the path ends `.xlsx` |
