"""Export the player profiles shown in the app (photos, stats, value history).

Usage:
    python -m src.export_profiles

No training: it reads the raw Transfermarkt tables and the player features
at SERVING_CUTOFF (the same catalogue as the served model) and writes:

- models/serving/player_profiles.parquet: one row per catalogue player with
  the photo URL, the stats of the year before the cutoff and their
  percentile among players of the same position (0 to 100), for the radar
- models/serving/value_history.parquet: market valuations of these players
  over the 5 years before the cutoff, for the value chart

The model and its embeddings are not touched.
"""

import sys

import pandas as pd

from src.clean_transfers import is_youth_or_reserve
from src.config import LEAGUES, SERVING_CUTOFF, SERVING_DIR
from src.features import _raw, load_or_build

# Stats shown on the radar, with the label used in the app.
RADAR = {
    "minutes": "Minutes played",
    "goals_per90": "Goals per 90",
    "assists_per90": "Assists per 90",
    "euro_games": "European games",
    "top_league_minutes_share": "Top-league minutes",
    "value_trend": "Value trend",
}
PROFILE_COLUMNS = [
    "player_id", "image_url", "foot", "height_cm", "games", "minutes", "goals_per90", "assists_per90",
    "cards_per90", "euro_games", "top_league_minutes_share", "value_trend", "peak_value_eur",
    "n_past_transfers", "club_id", "crest_club_id", "club_league_id",
]


NOT_A_CLUB = {"unknown", "without club", "retired", "career break"}


def crest_club_ids(features: pd.DataFrame, valuations: pd.DataFrame, min_share: float = 0.5,
                   min_count: int = 20) -> pd.Series:
    """Club id matching the club name shown for each player, for the crest.

    In some valuations the club name and the club id disagree (a player on
    loan at AS Roma carries Bayern's id). A club name gets the id it is most
    often paired with, but only when the name is unambiguous: at least
    `min_count` valuations and the same id in at least `min_share` of them
    ("AS Roma": 913 valuations with Roma's id, the rest spread over owners).
    Pseudo clubs ("Unknown") and ambiguous names keep the original id.
    """
    pairs = valuations.dropna(subset=["current_club_name", "current_club_id"])
    counts = pairs.groupby(["current_club_name", "current_club_id"]).size().reset_index(name="n")
    total = counts.groupby("current_club_name")["n"].transform("sum")
    counts["share"] = counts["n"] / total
    top = counts.sort_values("n").groupby("current_club_name").tail(1)
    top = top[(top["share"] >= min_share) & (total.loc[top.index] >= min_count)]
    top = top[~top["current_club_name"].str.lower().isin(NOT_A_CLUB)]
    best = top.set_index("current_club_name")["current_club_id"]
    return features["club_name"].map(best).fillna(features["club_id"])


GENERIC_WORDS = {"fc", "ac", "as", "sc", "cf", "cd", "ud", "sd", "rc", "afc", "club", "calcio", "sporting",
                 "football", "futbol", "de", "del", "la", "le", "1", "jk", "fk", "sk", "sv", "vfl", "vfb", "tsg",
                 "real", "united", "city", "athletic", "atletico", "racing", "olympique", "us", "ss", "ssc", "acf",
                 "associazione", "sportiva", "societa", "the"}


def _words(name) -> set[str]:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return {w for w in re.findall(r"[a-z]+", text) if w not in GENERIC_WORDS and len(w) > 2}


def same_club_name(shown: str, official: str) -> bool:
    """True if the displayed name and the clubs table name share a significant word."""
    return bool(_words(shown) & _words(official))


def club_leagues(crest_ids: pd.Series, original_ids: pd.Series, leagues: pd.Series, clubs: pd.DataFrame,
                 names: pd.Series | None = None) -> pd.Series:
    """League code of the displayed club: the crest club's league when the crest
    id was corrected and the club is in the clubs table, otherwise unchanged."""
    by_id = clubs.set_index("club_id")["domestic_competition_id"]
    fixed = crest_ids.map(by_id)
    use = (crest_ids != original_ids) & fixed.notna()
    if names is not None:  # a youth or reserve team keeps its own league, not its first team's
        use &= ~names.map(is_youth_or_reserve)
    return leagues.where(~use, fixed)


def build_profiles(features: pd.DataFrame, players: pd.DataFrame,
                   valuations: pd.DataFrame | None = None, clubs: pd.DataFrame | None = None) -> pd.DataFrame:
    f = features.copy()
    f["crest_club_id"] = f["club_id"]
    f["club_league_id"] = f["league_id"]
    if valuations is not None and clubs is not None:
        candidate = crest_club_ids(f, valuations)
        # only trust a correction that involves one of our 10 leagues (names like
        # "Alianza FC" are shared by clubs of several countries)
        by_id = clubs.set_index("club_id")
        cand_league = candidate.map(by_id["domestic_competition_id"])
        cand_name = candidate.map(by_id["name"])
        ok = cand_league.isin(LEAGUES) | f["league_id"].isin(LEAGUES)
        # the corrected club must look like the displayed one ("AS Roma" and
        # "Associazione Sportiva Roma"), not like the club owning most loanees
        ok &= pd.Series([same_club_name(a, b) if isinstance(b, str) else False
                         for a, b in zip(f["club_name"], cand_name)], index=f.index)
        f["crest_club_id"] = candidate.where(ok, f["club_id"])
        f["club_league_id"] = club_leagues(f["crest_club_id"], f["club_id"], f["league_id"], clubs, f["club_name"])
    info = players.set_index("player_id")[["image_url", "foot"]]
    f = f.join(info, on="player_id")
    out = f[PROFILE_COLUMNS].copy()
    out["foot"] = out["foot"].fillna("unknown")
    # percentile within the same position, so a defender is compared with defenders
    for col in RADAR:
        out[f"pct_{col}"] = (f.groupby("position")[col].rank(pct=True) * 100).round(0).fillna(0)
    return out


def build_value_history(valuations: pd.DataFrame, player_ids, cutoff: str, years: int = 5) -> pd.DataFrame:
    t = pd.Timestamp(cutoff)
    v = valuations[(valuations["date"] < t) & (valuations["date"] >= t - pd.Timedelta(days=int(365.25 * years)))]
    v = v[v["player_id"].isin(set(player_ids))]
    return (v[["player_id", "date", "market_value_in_eur"]]
            .rename(columns={"market_value_in_eur": "value_eur"})
            .sort_values(["player_id", "date"])
            .reset_index(drop=True))


def main() -> int:
    feats = load_or_build(SERVING_CUTOFF)
    served = pd.read_parquet(SERVING_DIR / "players.parquet", columns=["player_id"])
    feats = feats[feats["player_id"].isin(served["player_id"])]
    profiles = build_profiles(feats, _raw("players"), _raw("player_valuations"), _raw("clubs"))
    history = build_value_history(_raw("player_valuations"), profiles["player_id"], SERVING_CUTOFF)
    profiles.to_parquet(SERVING_DIR / "player_profiles.parquet", index=False)
    history.to_parquet(SERVING_DIR / "value_history.parquet", index=False)
    print(f"{len(profiles)} profiles, {len(history)} valuations -> {SERVING_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
