"""ScoutAI front end. Run with: streamlit run app/streamlit_app.py

Needs the API running (uvicorn api.main:app). Set SCOUTAI_API_URL if it
is not on http://localhost:8000.
"""

import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("SCOUTAI_API_URL", "http://localhost:8000")

st.set_page_config(page_title="ScoutAI", layout="wide")
st.title("ScoutAI")
st.caption("Player recommendations for your club, learned from 26,000 real transfers.")


@st.cache_data(ttl=600)
def get_clubs() -> pd.DataFrame:
    r = requests.get(f"{API}/clubs", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=600)
def get_defaults() -> dict:
    """Default weights of the API (tuned on validation), so the sliders start from them."""
    fallback = {"resale_weight": 0.5, "age_weight": 0.0, "mobility_weight": 0.0, "cold_slots": 0}
    try:
        r = requests.get(f"{API}/config", timeout=5)
        r.raise_for_status()
        return {**fallback, **r.json()}
    except requests.RequestException:
        return fallback


def log(club_id: int, player_id: int, action: str, rank: int, model_version: str | None) -> None:
    payload = {"club_id": club_id, "player_id": player_id, "action": action, "rank": rank,
               "model_version": model_version}
    try:
        requests.post(f"{API}/log_interaction", json=payload, timeout=5).raise_for_status()
        st.toast(f"{action.capitalize()} saved")
    except requests.RequestException as exc:
        st.error(f"Could not save the action: {exc}")


try:
    clubs = get_clubs()
except requests.RequestException:
    st.error(f"The API is not reachable on {API}. Start it with `uvicorn api.main:app`.")
    st.stop()
defaults = get_defaults()

labels = {f"{r.club_name} ({r.league_id})": r.club_id for r in clubs.itertuples()}
default = next((i for i, lab in enumerate(labels) if lab.startswith("Manchester United")), 0)

with st.sidebar:
    st.header("Your club")
    label = st.selectbox("Club", list(labels), index=default)
    club_id = labels[label]
    usual = float(clubs.set_index("club_id").loc[club_id, "budget_eur"])
    budget_m = st.slider("Budget per player (M EUR)", 0.5, 150.0, round(max(usual / 1e6, 0.5), 1), 0.5,
                         help="Defaults to what the club usually pays (90th percentile, last 3 years).")
    resale = st.slider("Resale focus", 0.0, 2.0, float(defaults["resale_weight"]), 0.1,
                       help="Favours players whose market value went up in the last 12 months.")
    youth = st.slider("Youth focus", 0.0, 2.0, float(defaults["age_weight"]), 0.1,
                      help="Pushes younger players up. The raw model favours players of 28-30, "
                           "clubs actually sign players of about 24.5.")
    mobility = st.slider("Penalty for frequent movers", 0.0, 2.0, float(defaults["mobility_weight"]), 0.1,
                         help="Pushes down players who changed club many times.")
    cold = st.slider("Places for new talents in the main list", 0, 3, min(int(defaults["cold_slots"]), 3),
                     help="0 gives the most accurate list. The new talents are also shown below.")
    position = st.selectbox("Position", ["All", "Goalkeeper", "Defender", "Midfield", "Attack"])
    k = st.slider("Number of players", 5, 20, 10)

params = {"club_id": club_id, "k": k, "budget_eur": budget_m * 1e6, "resale_weight": resale,
          "age_weight": youth, "mobility_weight": mobility, "cold_slots": cold}
if position != "All":
    params["position"] = position


def fetch(extra: dict) -> dict:
    resp = requests.get(f"{API}/recommend", params={**params, **extra}, timeout=10)
    if resp.status_code != 200:
        st.error(resp.json().get("detail", resp.text))
        st.stop()
    return resp.json()


def show(data: dict, section: str) -> None:
    for rec in data["recommendations"]:
        card(rec, data, section)


def card(rec: dict, data: dict, section: str) -> None:
    with st.container(border=True):
        left, mid, right = st.columns([4, 3, 2])
        tag = "  `new talent`" if rec["never_transferred"] else ""
        left.markdown(f"**{rec['rank']}. {rec['name']}**{tag}  \n"
                      f"{rec['sub_position'] if rec['sub_position'] != 'Missing' else rec['position']}, "
                      f"{rec['age']:.0f} y, {rec['country']}  \n{rec['current_club']} ({rec['league_id']})")
        trend = rec["value_trend_12m"]
        mid.metric("Market value", f"{rec['market_value_eur'] / 1e6:.1f} M EUR",
                   f"{(pow(2.718281828, trend) - 1) * 100:+.0f}% in 12 months")
        if rec["similar_to"]:
            sims = ", ".join(f"{s['name']} ({s['year']})" for s in rec["similar_to"])
            if rec["similar_to"][0].get("basis") == "profile":
                left.caption(f"Same profile as your signings: {sims}")
            else:
                left.caption(f"Similar to your signings: {sims}")
        if right.button("Shortlist", key=f"{section}s{rec['player_id']}"):
            log(club_id, rec["player_id"], "shortlisted", rec["rank"], data["model_version"])
        if right.button("Dismiss", key=f"{section}d{rec['player_id']}"):
            log(club_id, rec["player_id"], "dismissed", rec["rank"], data["model_version"])


data = fetch({})
st.subheader(f"Recommended for {data['club_name']}")
st.caption(f"Model {data['model_version']}, answered in {data['latency_ms']} ms. "
           f"Budget {data['budget_eur'] / 1e6:.1f} M EUR per player.")
show(data, "main")

st.subheader("New talents")
st.caption("Players with no recruitment by a club of our 10 leagues in the last 10 years, so unknown to the graph model. Ranked from their profile and stats by the cold-start encoder.")
show(fetch({"k": 5, "new_talents": "true"}), "new")

with st.expander("Shortlist and actions saved for this club"):
    r = requests.get(f"{API}/interactions", params={"club_id": club_id}, timeout=5)
    if r.ok and r.json():
        st.dataframe(pd.DataFrame(r.json()), hide_index=True)
    else:
        st.write("No action saved yet.")
