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
bet bet 1 --stake 25            # the id from the list
bet bet 1 --stake 25 --price 2.05   # if the price moved before you clicked
bet settle 1 won                # won | lost | push | void | half_won | half_lost
```

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
| **Pinnacle's own margin**, per outcome | Pinnacle sets margin inversely to limit as policy: ~2% on a game side it takes five figures on, 5–7% on a prop it takes a few hundred. A tight margin is Pinnacle telling you it is confident. |
| **Market tier** | Game sides and totals are where sharp money concentrates. Primary props are heavily bet but an order of magnitude thinner. Alternate lines are thinner again. |
| **Time to start** | Limits rise and prices converge as an event approaches. A prop posted Tuesday for a Sunday game is a placeholder. |

A fourth adjustment is about the maths rather than the market: on lopsided
prices the de-vig methods disagree most, so a fair probability near 0 or 1
carries more model risk regardless of liquidity. Anytime-touchdown and
similar yes/no longshots live here.

These combine into a 0–1 score used two ways:

- **A sliding EV bar.** Game lines flag at **+2%**, primary props at about
  **+3%**, alternate lines at about **+4.5%**. This is the Bayesian-correct
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

## Multi-leg tickets: pick'em and parlays

A different objective from everything above, and worth being precise about
because the difference determines the whole design.

The single-bet scanner looks for a soft book whose **price** beats
Pinnacle's de-vigged fair price on the same selection. The parlay
optimizer looks for a set of legs whose **joint probability** beats what
the payout structure assumes.

### Where the edge actually is

**Fixed-multiplier pick'em is the primary target.** Underdog and
PrizePicks set the payout by leg count and nothing else — a 2-pick pays
3x, a 5-pick pays 20x — and that multiple does not adjust for which legs
you chose. Two positively correlated legs hit together more often than a
3x multiple implies. The gap between the real joint probability and the
independence the multiple assumes is the edge, and it is structural: it is
there whether or not anybody made a mistake.

**DraftKings same-game parlays are the secondary, harder target.** DK runs
its own correlation model and already discounts correlated SGP legs. An
edge there means their correlation estimate is wrong, not merely that
correlation exists. The tool supports it, expects far fewer hits, and
flags every result from it as lower confidence. Pass `--offered-price`
with what the app actually shows, because the product of the legs is an
upper bound a same-game parlay will never pay.

**Cross-game parlays are a trap.** The vig compounds — four legs at 4.5%
hold each is `1.045⁴ − 1`, about 19% — and legs in different games have no
correlation to claw any of it back. The maths is implemented
(`copula.compounded_hold`) so the tool can state it with a number, and the
search refuses to build these by default.

### The marginals come from Pinnacle

This is the load-bearing idea. The pick'em sites post a line and a
multiplier; they do not post two sides, so there is no vig to strip and
nothing to de-vig there. The edge comes from having a better estimate of
each leg's probability than the pick'em site does — and that estimate is
Pinnacle's two-sided market on the same player, same stat, **same line**,
run through the existing `pricing.devig`.

Same line is not a detail. A leg compared against Pinnacle's number at a
different line is not a measurement of anything, so by default such a leg
is dropped. `--interpolate` will estimate one from Pinnacle's neighbouring
rungs instead — in probit space, refusing to extrapolate — and marks every
leg it touches as estimated and every ticket containing one as suspect.

**A leg with no Pinnacle reference is not usable**, and a ticket
containing one is not scored. That is a refusal, not a fallback.

### Joint probability: a Gaussian copula

No multiplying of marginals and no hand-waved "correlation bonus". Each
leg gets a latent standard normal oriented so large means the leg wins,
with its threshold set so `P(Z > t) = p`. The legs are drawn together from
a multivariate normal with correlation matrix `R`, and the joint
probability is the fraction of draws where every latent clears its
threshold.

A fixed seed makes the ranking reproducible; the standard error is
reported next to every estimate rather than hidden. If the assembled `R`
is not positive semi-definite — pairwise numbers from different sources
need not be mutually consistent — it is projected to the nearest PSD
matrix by eigenvalue clipping, and the ticket says so.

The same draws give the full distribution of *how many* legs landed, which
is what flex and insured entries actually pay on.

### Payout structures are data, not code

`betedge/data/payouts.yaml` holds a payout **vector** per leg count: what a
1-unit entry returns when exactly *k* legs hit, for `k = 0..n`. So

```
EV = Σₖ P(exactly k hit) × payout[k] − 1
```

Modelling only the all-hit case gets a flex entry badly wrong, so the full
vector is modelled. Pushes are modelled too: a leg landing exactly on an
integer line usually voids and shrinks the entry to the smaller table, and
integer lines on low-count stats are common. Where Pinnacle prices both
surrounding half-lines the push probability is measured exactly
(`P(X = L) = P(X > L−0.5) − P(X > L+0.5)`); where it does not, the
configured assumption is used and the leg is flagged.

> **The shipped ladders are unverified and every ticket says so.** They
> vary by state, change without notice, and an EV built on the wrong one is
> not slightly wrong — a 5-pick paying 10x instead of 20x turns a good bet
> into a bad one. Run `betedge parlay verify-payouts`, check it against
> your account, edit the file, set `verified: true`.

### Correlation: priors, then measurements — and always labelled

Two sources, in order of preference:

1. **Structural priors** (`betedge/data/correlation_priors.yaml`) —
   relationships true by construction. A quarterback's passing yards are
   the sum of his receivers' receiving yards (+0.45). Two backs split one
   pool of carries (−0.30). A goalie's saves are the other team's shots
   (+0.45). Every entry carries its reasoning in a comment. The **sign** of
   these is close to certain; the **magnitude** is judgement.
2. **Empirical estimates** fitted from game logs you supply. Fitted through
   Spearman rank correlation — immune to the long right tails of counting
   stats — and converted to the copula's latent scale with
   `ρ = 2 sin(π ρₛ / 6)`, which is exact for a Gaussian copula and assumes
   nothing about the marginals. Stored with the sample size, and used in
   place of the prior once there are enough joint observations
   (default 100).

Every pair reports which source it used, all the way through to the HTML
report. **An EV built entirely on priors is a hypothesis; one built on 500
games of joint data is an estimate.** They must never look alike.

Nothing is inferred from the odds. Backing a correlation out of DK's SGP
price would make this tool agree with DraftKings by construction, which
would guarantee it could never find the thing it is looking for.

### Rosters, which you should not have to maintain

The Odds API does not say which team a player plays for, and the sharpest
priors all need it — a QB with *his own* receiver is +0.45, two backs in
*the same* backfield are −0.30. Teams come from four layers, and you have
to do nothing for the first three:

1. **The game logs you already fit correlations from.** They carry a
   `team` column, because the fitter needs it to bucket pairs — it was
   simply being read and thrown away. `parlay correlations --from
   logs.csv` now keeps it, so the command you have to run anyway is the
   one that keeps your rosters current, and a mid-season trade corrects
   itself on the next refit.
2. **A published roster feed**, fetched during a scan once its snapshot
   has gone cold (`roster_refresh_days`, default 3). NFL is served by
   nflverse's open data release — a download, not a scrape. A failed
   fetch never stops a scan; it costs the sharp priors, not the run.
3. **Inference from the slate itself**, which needs no data at all. Some
   markets field exactly one player per team, so two of them in one game
   are necessarily opponents — the two starting quarterbacks, the two
   starting pitchers, the two goalies. Those labels are event-scoped and
   are never compared against a real club, because `evt#A` reading as
   "different team" from `KC` would be a confident answer nothing
   supports.
4. **Your own CSV** (`parlay.rosters_path`, or `--rosters`), which
   overrides all of them, because on the morning of a trade you know
   before any feed does.

```bash
betedge parlay rosters                 # coverage, sources, ages, conflicts
betedge parlay rosters --refresh       # pull the feed now
betedge parlay rosters --player "Patrick Mahomes"
```

**A stale roster is worse than no roster.** An unknown team costs you the
weak blended prior; a wrong team puts a confident +0.45 on a pair that is
really −0.10, and nothing downstream questions it. So every entry carries
the date it was true and where it came from, an entry past
`roster_max_age_days` is dropped rather than quietly used, two players
sharing a name are left unresolved rather than guessed between, and the
report names the club and its source on every leg.

That same reasoning rules out the tempting shortcut of having a language
model write the roster from memory: the answer would be fluent, undated,
and wrong about every transaction since its training cut-off. When this
was built, the live feed had Isiah Pacheco on Detroit — a model writing
from memory would have put him on Kansas City and turned an opponent into
a team mate.

### The honesty check that matters most

Every ticket reports its EV **with** the correlation matrix and **with** it
replaced by the identity — the latter computed exactly rather than
simulated, so the comparison is against a number with no noise in it. If a
ticket is positive only in the first, the entire case for betting it is a
correlation estimate rather than a price, and the report says so in a
coloured box rather than in a footnote.

### Guards

Same philosophy as `scan.py`: reject and flag rather than trust.

| Guard | What it does |
|---|---|
| EV above the plausibility ceiling | Suspect, no stake. A +40% pick'em ticket means a wrong line, a stale quote, or a payout table that doesn't match reality. |
| A leg with no Pinnacle reference | The ticket is not scored at all. |
| Positive only under assumed correlation | Flagged prominently, suspect, no stake. |
| Correlation entirely from priors | Flagged — a hypothesis, not a measurement. |
| Monte Carlo error large next to the edge | Flagged, suspect. |
| Same player twice | Rejected unless the product allows it. Two legs on the same prop are always rejected. |
| Legs on opposite sides of a correlated pair | Flagged as probable negative correlation, with the pair and the sign named. |
| Correlation matrix not PSD | Projected, and the projection is reported. |
| Cross-game parlay | Suspect, with its compounded hold stated. |
| Unverified payout table | Flagged on every ticket built from it. |

### Search and staking

Full enumeration is hopeless — 658,008 five-leg subsets of 40 candidates,
each needing its own simulation — so the search is a beam over leg sets,
deduplicated by leg set rather than order, with one shared pool of random
numbers so candidates are compared on the same draws. Finalists are
re-simulated at full precision. Results are returned ranked by EV **and**
separately by EV per unit of variance, because those are different
questions and blending them answers neither.

Staking reuses `pricing.kelly_fraction` on the price that reproduces the
ticket's EV, but a parlay breaks Kelly's assumptions harder than a single
bet: the payoff is lumpy, the probability error compounds across legs, and
correlated tickets lose their legs together. So the fraction is an eighth
of Kelly rather than a quarter, the per-ticket cap is tighter, and there is
a per-game exposure cap on top of the existing board-wide one — five
tickets on the same game are one bet, not five. The log-optimal fraction
computed straight off the Monte Carlo draws is reported alongside as a
cross-check.

Ranking is always by expected value. **A 20x ticket at −8% is a worse bet
than a 3x at +4%**, and nothing in the output is sorted in a way that says
otherwise.

### Profiles, so a recurring situation is one flag

A Thursday night game is a single event ~57 hours out, which the 48-hour
NFL prop window excludes — so a scan on the Tuesday finds nothing and
looks broken. Rather than edit the config and edit it back:

```bash
betedge profiles                        # what exists, and what each would change
betedge parlay scan --profile nfl-week  # apply one
```

`nfl-week` widens the look-ahead to 72 hours, drops the narrow
five-market override for the full registry list (13 markets, and a single
prime-time game costs 13 credits), and deepens the search since one game
is one group. Run it midweek and the 72-hour horizon isolates the Thursday
game on its own; run it Thursday or Friday and Sunday comes in with it.

**Every override is printed before the scan runs:**

```
Profile 'nfl-week' applied:
  sports                              [baseball_mlb, americanfootball_nfl] -> [americanfootball_nfl]
  prop_windows.americanfootball_nfl   48 -> 72
  prop_markets.americanfootball_nfl   5 items -> full registry list
  parlay.max_candidates_per_group     32 -> 48
```

That is not decoration. A profile can reach the staking fractions and the
guard thresholds, and a scan running under settings nobody stated is the
same class of failure as a stale roster — so it says what moved and what
it was before. A profile naming a setting that does not exist is an error
that names the valid ones, never a silent shrug.

Define your own under `profiles:` in `config.yaml`; a profile of the same
name replaces the shipped one. It may set `sports`, `core_sports`, the
per-sport `prop_markets` / `prop_windows` / `core_markets` maps (where
`null` removes your override and falls back to the registry), and fields
under `model:`, `parlay:`, `bankroll:` and `budget:`.

### Which sports

NBA, NFL, MLB and NHL — the leagues where Pinnacle and DraftKings both
carry deep player props. That is a starting point, not a conclusion:

```bash
betedge parlay coverage
```

probes the API and reports, per sport **and per book**, how many two-sided
Pinnacle prop markets exist and how many that book quotes **on the same
line** — the number that actually decides usability. A book can post a
thousand props and be worth nothing here if it prices them at numbers
Pinnacle does not touch, and only a per-book match rate shows that.

It asks about the known pick'em sites whether or not they are configured,
because cost is `markets × ceil(books ÷ 10)` — ten books bill as one — so
the honest way to find out whether a book is available to you is to ask
for it and report what came back. Books that are asked for and never
answer are named, rather than left as an absence in a table. Tennis, MMA and soccer are
excluded from the prop-based optimizer with the reason stated in
`parlay.PROP_OPTIMIZER_EXCLUDED`: Pinnacle prices almost no player props
in them, so there is nothing to build a marginal from. Their edge is in
core markets, which the single-bet scanner already covers.

### Ticket closing-line value

`betedge close` re-prices every leg of every open ticket at Pinnacle's
closing number and pushes the result back through the same copula, so a
ticket gets a joint probability at the close to compare against the
modelled one. Tickets that were never bet are captured too — they are most
of the evidence. This is the only fast read on whether any of this works:
profit and loss on parlays is so noisy that a hundred settled entries still
tell you nothing.

---

## Layout

```
betedge/
  pricing.py    de-vig, EV, Kelly          ← the maths, heavily tested
  copula.py     Gaussian copula: the joint probability of a multi-leg ticket
  correlation.py structural priors, fitted estimates, matrix assembly
  rosters.py    who plays for whom, and how much to trust that answer
  parlay.py     pick'em and parlay tickets: legs, guards, search, staking
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
  data/
    payouts.yaml             pick'em payout ladders — YOU must verify these
    correlation_priors.yaml  structural correlation priors, with reasoning
tests/          656 tests; no network, no credits spent
                (enforced: requests is blocked for the whole suite)
```

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Commands

| Command | Cost | What it does |
|---|---|---|
| `bet profiles` | free | Named override bundles, and what each would change against your config |
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
| `bet parlay verify-payouts` | free | Print the payout ladders. **Read this before trusting any ticket EV.** |
| `bet parlay coverage` | ~1/event | Which sports have usable two-sided Pinnacle prop coverage |
| `bet parlay correlations` | free | Fit pairwise correlations from your game logs — and learn rosters from the same file |
| `bet parlay rosters` | free | Who plays for whom, where it came from, how stale it is. `--refresh` pulls the feed |
| `bet parlay scan` | budgeted | Build and rank multi-leg tickets |
| `bet parlay bet <id> --stake N` | free | Log an entry you placed |
| `bet parlay settle <id> --hit K` | free | Settle it by how many legs landed |
