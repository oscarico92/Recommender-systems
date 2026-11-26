# Systems Engineer: app checks and demo notes

Scope: Hugo Bassaget — FastAPI, Streamlit, running the app and the README.

## Starting point

The API, page and real model were already integrated by the team in PR #1. PR #4 added the feedback mechanism, and PR #8 integrated the week 4 model changes. PR #9 improves the main-list explanations. This system pass builds on that work; it does not introduce a new model or retrain it.

The additional changes are in the page: refresh recommendations after an action, show over-budget alternatives, explain the comparison labels, and handle unavailable API responses. The tests cover the API and the page together with the committed model and a temporary database.

## How a request works

1. Streamlit gets the clubs from `/clubs` and the tuned weights from `/config`.
2. The controls are sent to `/recommend`: club, budget, position and re-ranking weights.
3. FastAPI loads the exported model once and keeps it in memory. `src/predict.py` scores players and applies the shared re-ranking code.
4. The page displays a main list and a separate new-talents list.
5. Shortlist or Dismiss calls `/log_interaction`. SQLite stores the club, player, action, rank, model version and time.
6. After a successful save, Streamlit reruns and fetches both lists. `/recommend` excludes saved players and updates the club vector using the feedback. There is no retraining during a click.

| File | What to be able to explain |
| --- | --- |
| `app/streamlit_app.py` | Controls, HTTP requests, cards, errors and refresh after a click |
| `api/main.py` | `/clubs`, `/config`, `/recommend`, validation and feedback readback |
| `api/interactions.py`, `src/db.py` | Simon's interaction endpoint and SQLite storage, used by the page |
| `src/predict.py` | The team's prediction and explanation functions called by the API |
| `src/rerank.py` | Budget filtering and the age/resale/mobility adjustments |

## Local checks, 21 November 2026

Tested on macOS with Python 3.11.9 in a new virtual environment, using `requirements.txt` and the committed `models/serving/`. No raw dataset download or training is needed for this path.

```bash
python -m pytest -q
```

Result: **56 passed, 1 skipped**. The skipped test needs PyTorch and checks graph weights; it is outside the lightweight serving environment. The API tests check position filters, new talents, budget flags, invalid requests and feedback isolation between clubs. The page tests exercise successful and failed clicks, missing API responses and the model's slider defaults.

The new page tests initially reproduced six failures: the two action buttons left stale cards, `/config` silently used fallback settings, unavailable recommendations/history produced exceptions, and over-budget cards were unmarked. These cases now pass.

Browser check with the real local API: Shortlist saved one action for Dean Huijsen, Dismiss saved one for Enzo Barrenechea. Each player disappeared from the main list after the click; `/interactions` returned exactly the two actions. These clicks used a separate rehearsal database.

HTTP timings after model loading, 20 requests per club, including the local HTTP round trip:

| Club | Median | Maximum |
| --- | --- | --- |
| Manchester United | 11.05 ms | 19.72 ms |
| CA Osasuna | 11.34 ms | 12.58 ms |
| Paris Saint-Germain | 10.67 ms | 11.21 ms |

This checks the under-two-second target locally. It is not a load test or a measurement of a remote deployment.

### PR #9 check

Compared `main` at `5060ab3` with PR #9 at `d60fa7f`, using the same exported model and final API defaults: resale 1, age 2, mobility 0. Across 322 clubs and 3,220 recommendations:

- Same-position explanations: **24.35% → 95.05%**.
- Distinct explanation pairs within each top 10: **74.19% → 70.00%**.
- Player order, scores and other recommendation fields are unchanged.
- Empty history, no matching position and only one matching signing have a valid fallback.

The percentages differ slightly from the PR description because this check uses the final model settings. The result is the same trade-off: more relevant positions, slightly fewer distinct pairs. GitHub still needs to target `main` before merging #9; it currently targets Mathis's integration branch.

## Two-minute demo

Start the API and page using the README. Keep the exported model; retraining can change the players shown. For a rehearsal, use a separate SQLite file so earlier clicks do not hide the demo players.

Before the demo, create the accounts of the three case-study clubs (Manchester United, Benfica, Celtic) on the login page, so switching club is "Log out" then "Sign in".

1. Sign in with the Manchester United account (or create it live: username, password, recruitment club). Point out that the sliders start from the settings selected on validation.
2. Explain one recommendation and its reference signings. The position is prioritised; the similarity inside that group comes from the embeddings.
3. Lower the budget and show an "Above budget" label if the candidate list needs alternatives. Market value is an estimate, not a guaranteed price.
4. Open the new-talents section. Explain that "new" means unknown to this transfer graph, not necessarily young or unknown in football.
5. Shortlist a player. The card disappears and the list updates. Open the action history to show the saved row.
6. Show `/docs` and one `/recommend` request if asked about the backend.

## Questions to practise

**Why FastAPI and Streamlit?** They let us connect a Python model to a usable demo with little setup. FastAPI validates the request parameters; Streamlit handles the controls and display.

**Why no PyTorch when launching the app?** Training exports embeddings and tables. Serving only needs NumPy and pandas to score and format the results.

**Why rerun after a click?** Streamlit had already fetched the recommendations before handling the button. Saving the action alone did not refresh those cards. Rerunning reads the updated feedback before displaying the lists.

**Does feedback improve accuracy?** It adapts the list to the scout's choices. The team's offline check did not show an accuracy gain, so we do not claim one.

**Are the reference players the model's reason?** No. They are comparisons: same position first for the main list, profile comparisons for new talents. They help inspect the suggestions without proving a causal explanation.

**Why can a player exceed the budget?** The existing re-ranking completes a short list with more expensive candidates. The page now makes that visible. Changing it into a strict cap would be a separate model/product decision.

**Is this a production service?** No. This is a local showcase app with SQLite and no user accounts. The checks demonstrate the requested end-to-end flow, not multi-user production readiness.
