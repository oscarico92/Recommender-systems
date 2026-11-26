"""ScoutAI front end. Run with: streamlit run app/streamlit_app.py

Needs the API running (uvicorn api.main:app). Set SCOUTAI_API_URL if it
is not on http://localhost:8000.

Photos and club crests are loaded by the viewer's browser from
Transfermarkt. Without internet the cards show the player's initials.
"""

import html
import math
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("SCOUTAI_API_URL", "http://localhost:8000")
ASSETS = Path(__file__).parent / "assets"
GREEN, GOLD, INK, MUTED = "#0B6E4F", "#F2C94C", "#1B2430", "#5B6573"
RADAR_LABELS = {
    "minutes": "Minutes played",
    "goals_per90": "Goals per 90",
    "assists_per90": "Assists per 90",
    "euro_games": "European games",
    "top_league_minutes_share": "Top-league minutes",
    "value_trend": "Value trend",
}

st.set_page_config(page_title="ScoutAI", page_icon=str(ASSETS / "icon.png"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), size="large")
st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; }}
  .sa-head {{ display: flex; align-items: center; gap: 14px; margin-bottom: 2px; }}
  .sa-head img {{ height: 52px; }}
  .sa-head h1 {{ font-size: 1.9rem; margin: 0; padding: 0; color: {INK}; }}
  .sa-tag {{ color: {MUTED}; font-size: 0.95rem; margin: 0 0 6px 0; }}
  .sa-crest {{ width: 46px; height: 46px; object-fit: contain; }}
  .sa-avatar {{ position: relative; width: 64px; height: 64px; border-radius: 50%; flex: none;
               background: {GREEN}; color: #fff; font-weight: 700; font-size: 1.15rem;
               display: flex; align-items: center; justify-content: center; overflow: hidden;
               box-shadow: 0 0 0 3px #fff, 0 0 0 4px #d9dee5; }}
  .sa-avatar .sa-photo {{ position: absolute; inset: 0; background-size: cover;
                          background-position: center top; }}
  .sa-row {{ display: flex; gap: 12px; align-items: center; }}
  .sa-name {{ font-weight: 700; font-size: 1.05rem; color: {INK}; line-height: 1.2; }}
  .sa-rank {{ display: inline-block; min-width: 24px; padding: 1px 6px; margin-right: 6px; border-radius: 10px;
              background: {GREEN}; color: #fff; font-size: 0.78rem; text-align: center; }}
  .sa-new {{ display: inline-block; padding: 1px 7px; margin-left: 6px; border-radius: 10px;
             background: {GOLD}; color: {INK}; font-size: 0.72rem; font-weight: 700; }}
  .sa-sub {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.35; }}
  .sa-club {{ display: flex; align-items: center; gap: 6px; color: {INK}; font-size: 0.86rem; }}
  .sa-club img {{ width: 18px; height: 18px; object-fit: contain; }}
  .sa-value {{ font-size: 1.35rem; font-weight: 700; color: {INK}; }}
  .sa-up {{ color: {GREEN}; font-weight: 600; font-size: 0.85rem; }}
  .sa-kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 6px; }}
  .sa-kpi {{ background: #F3F6F8; border-radius: 8px; padding: 8px 10px; }}
  .sa-kpi-l {{ color: {MUTED}; font-size: 0.75rem; }}
  .sa-kpi-v {{ color: {INK}; font-size: 1.1rem; font-weight: 700; white-space: nowrap; }}
  .sa-down {{ color: #B3261E; font-weight: 600; font-size: 0.85rem; }}
</style>
""", unsafe_allow_html=True)

if "saved_action" in st.session_state:
    st.toast(st.session_state.pop("saved_action"))


# ---------- API calls ----------

@st.cache_data(ttl=600)
def get_clubs() -> pd.DataFrame:
    r = requests.get(f"{API}/clubs", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=600)
def get_defaults() -> dict:
    """Default weights of the API (tuned on validation), so the sliders start from them."""
    r = requests.get(f"{API}/config", timeout=5)
    r.raise_for_status()
    return r.json()


def get_profiles(ids: list[int]) -> dict[int, dict]:
    """Photos, stats and value history. The page still works without them."""
    unique = list(dict.fromkeys(int(i) for i in ids))
    profiles: dict[int, dict] = {}
    # The API serves at most 20 players per call, so load them in batches.
    for start in range(0, len(unique), 20):
        batch = unique[start:start + 20]
        try:
            r = requests.get(f"{API}/players", params={"ids": ",".join(str(i) for i in batch)}, timeout=10)
            r.raise_for_status()
            profiles.update({p["player_id"]: p for p in r.json()})
        except (requests.RequestException, ValueError):
            continue
    return profiles


def log(club_id: int, player_id: int, action: str, rank: int | None, model_version: str | None) -> None:
    payload = {"club_id": club_id, "player_id": player_id, "action": action, "rank": rank,
               "model_version": model_version}
    try:
        requests.post(f"{API}/log_interaction", json=payload, timeout=5).raise_for_status()
    except requests.RequestException:
        st.error("Could not confirm the action. Check the saved actions before trying again.")
        return
    # The cards were fetched before the click. Fetch both lists again so the
    # saved player disappears and the club's feedback is applied straight away.
    st.session_state["saved_action"] = f"{action.capitalize()} saved, recommendations updated"
    st.rerun()


# ---------- small display helpers ----------

def esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def initials(name: str) -> str:
    parts = [p for p in str(name).replace("-", " ").split() if p]
    return esc("".join(p[0] for p in parts[:2]).upper() or "?")


def avatar(name: str, photo_url: str | None, size: int = 64) -> str:
    """Photo over the initials: if the photo cannot load, the initials stay visible."""
    photo = f'<div class="sa-photo" style="background-image:url(\'{esc(photo_url)}\')"></div>' if photo_url else ""
    return (f'<div class="sa-avatar" style="width:{size}px;height:{size}px">'
            f'{initials(name)}{photo}</div>')


def crest(url: str | None, size: int = 18) -> str:
    """Club crest as a background image: nothing is shown if it cannot load (no broken icon)."""
    if not url:
        return ""
    return (f'<span style="display:inline-block;width:{size}px;height:{size}px;flex:none;'
            f'background:url(\'{esc(url)}\') center/contain no-repeat"></span>')


def trend_html(trend: float) -> str:
    pct = (math.exp(trend) - 1) * 100
    cls = "sa-up" if pct >= 0.5 else ("sa-down" if pct <= -0.5 else "sa-sub")
    arrow = "&#9650;" if pct >= 0.5 else ("&#9660;" if pct <= -0.5 else "&#8226;")
    return f'<span class="{cls}">{arrow} {pct:+.0f}% in 12 months</span>'


def position_label(p: dict) -> str:
    return p["sub_position"] if p.get("sub_position") not in (None, "Missing") else p["position"]


def radar(players: list[dict]) -> go.Figure:
    fig = go.Figure()
    labels = list(RADAR_LABELS.values())
    colors = [GREEN, GOLD, "#3D5A80"]
    for i, p in enumerate(players):
        pct = p.get("percentiles") or {}
        r = [float(pct.get(k) or 0) for k in RADAR_LABELS]
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=labels + labels[:1], fill="toself", name=p["name"],
                                      line=dict(color=colors[i % 3], width=2), opacity=0.75))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], showticklabels=False),
                                 angularaxis=dict(tickfont=dict(size=11))),
                      showlegend=len(players) > 1, margin=dict(l=90, r=90, t=30, b=30), height=320,
                      legend=dict(orientation="h", y=-0.15), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def value_chart(player: dict) -> go.Figure | None:
    hist = player.get("value_history") or []
    if len(hist) < 2:
        return None
    df = pd.DataFrame(hist)
    fig = go.Figure(go.Scatter(x=pd.to_datetime(df["date"]), y=df["value_eur"] / 1e6, mode="lines+markers",
                               line=dict(color=GREEN, width=2), marker=dict(size=5),
                               hovertemplate="%{x|%b %Y}: %{y:.1f} M EUR<extra></extra>"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="M EUR",
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def scouting_report(p: dict) -> None:
    """Radar of percentiles within the position, value curve and key stats."""
    if not p.get("percentiles"):
        st.caption("No detailed statistics for this player.")
        return
    st.caption("Percentile among players of the same position (last 12 months)")
    st.plotly_chart(radar([p]), use_container_width=True, config={"displayModeBar": False},
                    key=f"radar{p['player_id']}{st.session_state.setdefault('_chart', 0)}")
    st.session_state["_chart"] += 1
    st.caption("Market value over the last 5 years")
    fig = value_chart(p)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                        key=f"value{p['player_id']}{st.session_state['_chart']}")
        st.session_state["_chart"] += 1
    s = p.get("stats") or {}
    st.markdown(
        f"**{s.get('games') or 0:.0f}** games, **{s.get('minutes') or 0:.0f}** minutes  \n"
        f"**{s.get('goals_per90') or 0:.2f}** goals and **{s.get('assists_per90') or 0:.2f}** assists per 90  \n"
        f"**{s.get('euro_games') or 0:.0f}** European games, {s.get('height_cm') or 0:.0f} cm, "
        f"{s.get('foot') or 'unknown'} foot  \n"
        f"Peak value **{(s.get('peak_value_eur') or 0) / 1e6:.1f} M EUR**, "
        f"{s.get('n_past_transfers') or 0:.0f} past transfers")


# ---------- page setup ----------

try:
    clubs = get_clubs()
    defaults = get_defaults()
except requests.RequestException:
    st.error(f"Could not load the clubs and model settings from {API}. Check that the API is running.")
    st.button("Retry connection")
    st.stop()

labels = {f"{r.club_name} ({r.league_id})": r.club_id for r in clubs.itertuples()}
default = next((i for i, lab in enumerate(labels) if lab.startswith("Manchester United")), 0)

with st.sidebar:
    st.header("Your club")
    label = st.selectbox("Club", list(labels), index=default)
    club_id = labels[label]
    usual = float(clubs.set_index("club_id").loc[club_id, "budget_eur"])
    budget_m = st.slider("Budget per player (M EUR)", 0.05, 150.0, round(max(usual / 1e6, 0.05), 2), 0.05,
                         help="Based on the market values of the club's signings (90th percentile, last 3 years). "
                              "These are estimated values, not transfer fees.")
    position = st.selectbox("Position", ["All", "Goalkeeper", "Defender", "Midfield", "Attack"])
    k = st.slider("Number of players", 5, 20, 10)
    st.subheader("Strategy")
    resale = st.slider("Resale focus", 0.0, 2.0, float(defaults["resale_weight"]), 0.1,
                       help="Favours players whose market value went up in the last 12 months.")
    youth = st.slider("Youth focus", 0.0, 2.0, float(defaults["age_weight"]), 0.1,
                      help="Pushes younger players up. The raw model favours players of 28-30, "
                           "clubs actually sign players of about 24.5.")
    mobility = st.slider("Penalty for frequent movers", 0.0, 2.0, float(defaults["mobility_weight"]), 0.1,
                         help="Pushes down players who changed club many times.")
    cold = st.slider("Places for new talents in the main list", 0, 3, min(int(defaults["cold_slots"]), 3),
                     help="0 gives the most accurate list. The new talents also have their own tab.")
    st.caption(f"Model {defaults.get('model_version', '')}. Defaults tuned on the 2024/25 season.")
    st.caption("Data: transfermarkt-datasets (CC0). Photos and crests: Transfermarkt. "
               "Photos and crests need an internet connection.")

params = {"club_id": club_id, "k": k, "budget_eur": budget_m * 1e6, "resale_weight": resale,
          "age_weight": youth, "mobility_weight": mobility, "cold_slots": cold}
if position != "All":
    params["position"] = position


def fetch(extra: dict) -> dict:
    try:
        resp = requests.get(f"{API}/recommend", params={**params, **extra}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        st.error("Could not load recommendations. Check the API connection and try again.")
        st.button("Retry recommendations")
        st.stop()


def card(rec: dict, data: dict, section: str, profile: dict | None) -> None:
    with st.container(border=True):
        new = '<span class="sa-new">NEW TALENT</span>' if rec["never_transferred"] else ""
        st.markdown(
            f'<div class="sa-row">{avatar(rec["name"], rec.get("photo_url"))}<div>'
            f'<div class="sa-name"><span class="sa-rank">{rec["rank"]}</span>{esc(rec["name"])}{new}</div>'
            f'<div class="sa-sub">{esc(position_label(rec))}, {rec["age"]:.0f} y, {esc(rec["country"])}</div>'
            f'<div class="sa-club">{crest(rec.get("club_logo_url"))}{esc(rec["current_club"])} '
            f'({esc(rec["league_id"])})</div>'
            f'</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sa-value">{rec["market_value_eur"] / 1e6:.1f} M EUR</div>'
                    f'{trend_html(rec["value_trend_12m"])}', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("Shortlist", key=f"{section}s{rec['player_id']}", type="primary", use_container_width=True):
            log(club_id, rec["player_id"], "shortlisted", rec["rank"], data["model_version"])
        if b2.button("Dismiss", key=f"{section}d{rec['player_id']}", use_container_width=True):
            log(club_id, rec["player_id"], "dismissed", rec["rank"], data["model_version"])
        if not rec["within_budget"]:
            st.warning("Above budget, shown to complete the list")
        if rec["similar_to"]:
            sims = ", ".join(f"{s['name']} ({s['year']})" for s in rec["similar_to"])
            if rec["similar_to"][0].get("basis") == "profile":
                st.caption(f"Profile comparison with your signings: {sims}")
            else:
                st.caption(f"Recruitment similarity (same position first): {sims}")
        with st.expander("Scouting report"):
            scouting_report(profile or {})


def show(data: dict, section: str, profiles: dict, columns: int = 2) -> None:
    items = data["recommendations"]
    if not items:
        st.info("No players found for these settings. Try another position or club.")
        return
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, rec in zip(cols, items[start:start + columns]):
            with col:
                card(rec, data, section, profiles.get(rec["player_id"]))


def table(data: dict) -> None:
    rows = [{
        "Photo": r.get("photo_url"), "Rank": r["rank"], "Player": r["name"], "Position": position_label(r),
        "Age": round(r["age"]), "Crest": r.get("club_logo_url"), "Club": r["current_club"],
        "Value (M EUR)": r["market_value_eur"] / 1e6, "12-month trend": (math.exp(r["value_trend_12m"]) - 1) * 100,
        "Within budget": r["within_budget"], "New talent": r["never_transferred"],
    } for r in data["recommendations"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, row_height=52, column_config={
        "Photo": st.column_config.ImageColumn(width="small"),
        "Crest": st.column_config.ImageColumn(width="small"),
        "Value (M EUR)": st.column_config.NumberColumn(format="%.1f"),
        "12-month trend": st.column_config.NumberColumn(format="%+.0f%%"),
    })
    st.caption("Shortlist and Dismiss are on the cards view.")


# ---------- content ----------

data = fetch({})
talents = fetch({"k": 6, "new_talents": "true"})
profiles = get_profiles([r["player_id"] for r in data["recommendations"] + talents["recommendations"]])

head_left, head_right = st.columns([5, 4])
with head_left:
    st.markdown(f'<div class="sa-head">{crest(data.get("club_logo_url"), 48)}<h1>{esc(data["club_name"])}</h1></div>'
                f'<p class="sa-tag">Recruitment recommendations learned from 26,000 real transfers '
                f'of 322 European clubs.</p>', unsafe_allow_html=True)
with head_right:
    fb = data.get("feedback", {})
    kpis = [("Budget per player", f"{data['budget_eur'] / 1e6:.1f} M EUR"), ("Shortlisted", fb.get("liked", 0)),
            ("Dismissed", fb.get("dismissed", 0)), ("Response time", f"{data['latency_ms']:.0f} ms")]
    st.markdown('<div class="sa-kpis">' + "".join(
        f'<div class="sa-kpi"><div class="sa-kpi-l">{esc(lab)}</div><div class="sa-kpi-v">{esc(val)}</div></div>'
        for lab, val in kpis) + "</div>", unsafe_allow_html=True)

tab_main, tab_new, tab_short = st.tabs(["Recommendations", "New talents", "Shortlist and compare"])

with tab_main:
    view = st.radio("View", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")
    with st.expander("How to read the recommendations"):
        st.write("The main list follows the club's past recruitment patterns. The examples favour "
                 "past signings in the same position, then use recruitment similarity. "
                 "New-talent examples compare position and age. These comparisons help you read "
                 "the suggestions; they do not prove why a player should be signed.")
        st.write("Market values are estimates. When too few candidates fit the budget, "
                 "the list includes clearly marked alternatives above it. Shortlist and Dismiss "
                 "save your choice, hide that player and update the recommendations. "
                 "The scouting report compares each statistic with players of the same position.")
    if view == "Cards":
        show(data, "main", profiles)
    else:
        table(data)

with tab_new:
    st.caption("Players with no recruitment by a club of our 10 leagues in the last 10 years, so unknown "
               "to the graph model. Ranked from their profile and statistics by the cold-start encoder.")
    show(talents, "new", profiles, columns=3)

with tab_short:
    try:
        r = requests.get(f"{API}/interactions", params={"club_id": club_id, "limit": 1000}, timeout=5)
        r.raise_for_status()
        actions = r.json()
    except requests.RequestException:
        actions = None
        st.warning("Saved actions could not be loaded. They may still be in the database.")
        st.button("Retry history")

    if actions is not None:
        last: dict[int, str] = {}
        for a in sorted(actions, key=lambda a: a["id"]):
            if a["action"] != "viewed":
                last[a["player_id"]] = a["action"]
        short_ids = [p for p, a in last.items() if a in ("shortlisted", "signed")]
        short = get_profiles(short_ids)

        st.subheader(f"Your shortlist ({len(short_ids)})")
        if not short_ids:
            st.info("Shortlist players from the recommendations to compare them here.")
        cols = st.columns(4)
        for i, pid in enumerate(short_ids):
            p = short.get(pid, {"name": f"Player {pid}", "player_id": pid})
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="sa-row">{avatar(p["name"], p.get("photo_url"), 48)}<div>'
                        f'<div class="sa-name">{esc(p["name"])}</div>'
                        f'<div class="sa-sub">{esc(p.get("position", ""))}, {p.get("age", 0):.0f} y</div>'
                        f'<div class="sa-sub">{(p.get("market_value_eur") or 0) / 1e6:.1f} M EUR</div>'
                        f'</div></div>', unsafe_allow_html=True)

        st.subheader("Compare players")
        pool = {**{p: q for p, q in profiles.items()}, **short}
        names = {f"{q['name']} ({q.get('current_club', '')})": p for p, q in pool.items()}
        default_pick = [n for n, p in names.items() if p in short_ids][:3]
        if len(default_pick) < 2:
            default_pick = list(names)[:2]
        picked = st.multiselect("Choose 2 or 3 players (shortlist or current recommendations)", list(names),
                                default=default_pick, max_selections=3)
        chosen = [pool[names[n]] for n in picked]
        if len(chosen) >= 2:
            st.plotly_chart(radar(chosen), use_container_width=True, config={"displayModeBar": False},
                            key="compare_radar")
            rows = {}
            for q in chosen:
                s = q.get("stats") or {}
                rows[q["name"]] = {
                    "Position": q.get("position"), "Age": q.get("age"),
                    "Club": q.get("current_club"),
                    "Value (M EUR)": round((q.get("market_value_eur") or 0) / 1e6, 1),
                    "Minutes (12 m)": s.get("minutes"), "Goals / 90": round(s.get("goals_per90") or 0, 2),
                    "Assists / 90": round(s.get("assists_per90") or 0, 2),
                    "European games": s.get("euro_games"),
                }
            # one row per player, so every column has a single type
            st.dataframe(pd.DataFrame.from_dict(rows, orient="index"), use_container_width=True)
        else:
            st.caption("Pick at least 2 players.")

        with st.expander("Shortlist and actions saved for this club"):
            if actions:
                st.dataframe(pd.DataFrame(actions), hide_index=True)
            else:
                st.write("No action saved yet.")
