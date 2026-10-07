# DECISIONS.md

End date: ____  (write it before anything else)

One line per decision: what was chosen, why, what was rejected. The first
section was made by the starter kit. Read it, disagree where you like, and
change the constant in `src/recsys/config.py` if you do. Then log the change.

## Made by the starter kit (Steps 1–4)

**D1. Dataset: MovieLens, developed on `ml-latest-small`, final numbers on `ml-32m`.**
Small runs in seconds; 32M is the stable benchmark release. Rejected: Netflix
Prize data (titles only, no links to plot text).

**D2. Text source: TMDB via MovieLens `links.csv`.** Fields cached: overview,
tagline, keywords, genres, directors, top 5 cast, runtime, language, release
date, vote count/average, popularity. Fetched most-rated first, so stopping
early still covers most ratings. Rejected: IMDb datasets (no plot text in the
free dumps), scraping (terms of use).

**D3. TMDB popularity fields are cached but are a leakage risk.** `vote_count`,
`vote_average` and `popularity` reflect TMDB *today*, after every MovieLens
timestamp. Using them as features or tie-breakers lets models see the future.
Fine for display; think twice before ranking on them.

**D4. Sessions: a gap of more than 30 minutes starts a new session.** MovieLens
users bulk-rate dozens of films at signup, often with identical timestamps, so
"the last film rated" is frequently an arbitrary pick from a batch. Sessions
make "what came next" mean the next *visit*. Rejected: dropping signup ratings
(deletes most users), raw timestamp order (ties make the target arbitrary).

**D5. Positives: rating ≥ 4.0.** Targets are films the user liked, not merely
rated. All ratings (including low ones) stay in the history, so you can choose
whether models use them. Rejected: every rating as positive (a 1-star film is
a bad "next recommendation" target).

**D6. Split: per-user, session-aware leave-last-out.** Test target = one random
positive from the user's last session containing a positive; validation target
= one from the positive session before that. Train ends before the val session;
train_full ends before the test session. Other ratings from the test session
are dropped so same-session co-ratings can't leak.

**D7. Eligibility: users need two positive sessions and at least 5 positives
before the validation session.** Others aren't scored but all their ratings are
still used for training. Expect a large share of ml-32m users to be ineligible:
many only ever use the site once. That's a property of the data worth one line
in the write-up.

**D8. Known limitation: no global time cutoff.** User A's history may be later
in calendar time than user B's test target, so CF models trained on everyone
can "see the future" for B. Common in the literature, but it flatters CF
slightly. Rejected for now for simplicity; revisit if CF numbers look
suspiciously good.

**D9. Popularity buckets from training data only.** Head = most-rated films
covering 80% of training interactions; tail = the rest with at least one; cold
= none. Computed separately for val (from train) and test (from train_full).

**D10. No film filtering.** Rarely rated films are exactly the long tail the
hybrid is meant to win on. Users can be subsampled (`--max-users`) for speed.

**D11. Seed 42 everywhere; the split is generated once and frozen.**

## Yours (Step 5 onward)

- Eval: k values, exclusion of already-seen films, full-catalogue ranking
- Embedding text: which fields, in what template
- Embedding model:
- Vector store:
- User profile vector:
- Hybrid blend:
- Query set scoring rule:
