from datetime import date

import pandas as pd

from src.clean_transfers import clean_transfers, is_youth_or_reserve, parent_name

COLS = ["player_id", "transfer_date", "transfer_season", "from_club_id", "to_club_id",
        "from_club_name", "to_club_name", "transfer_fee", "market_value_in_eur", "player_name"]


def t(pid, d, f, t_, fn, tn, fee):
    return [pid, d, "", f, t_, fn, tn, fee, 1e6, f"p{pid}"]


CLUBS = pd.DataFrame({
    "club_id": [1, 2, 3, 9],
    "name": ["Alpha FC", "Beta FC", "Gamma FC", "Outside FC"],
    "domestic_competition_id": ["FR1", "FR1", "GB1", "XX1"],
})


def run(rows, min_club=1):
    df = pd.DataFrame(rows, columns=COLS)
    return clean_transfers(df, CLUBS, leagues={"FR1": "", "GB1": ""},
                           min_club_transfers=min_club, today=date(2026, 7, 6))


def test_name_helpers():
    assert is_youth_or_reserve("Manchester United U21")
    assert is_youth_or_reserve("Real Madrid Castilla")
    assert is_youth_or_reserve("FC Porto B")
    assert not is_youth_or_reserve("Club Brugge KV")
    assert parent_name("FC Porto B") == parent_name("FC Porto")


def test_permanent_transfer_kept():
    inter, _, _ = run([t(10, "2020-07-01", 2, 1, "Beta FC", "Alpha FC", 5e6)])
    assert len(inter) == 1
    row = inter.iloc[0]
    assert row.club_id == 1 and not row.is_loan and not row.is_free


def test_loan_flagged_and_return_dropped():
    inter, _, rep = run([
        t(11, "2020-08-01", 1, 2, "Alpha FC", "Beta FC", 0),   # loan
        t(11, "2021-06-30", 2, 1, "Beta FC", "Alpha FC", 0),   # end of loan
    ])
    assert len(inter) == 1
    assert inter.iloc[0].club_id == 2 and inter.iloc[0].is_loan
    assert rep["n_loans"] == 1


def test_long_gap_is_not_a_loan():
    inter, _, _ = run([
        t(12, "2015-08-01", 1, 2, "Alpha FC", "Beta FC", 0),
        t(12, "2020-08-01", 2, 1, "Beta FC", "Alpha FC", 0),
    ])
    assert len(inter) == 2 and not inter.is_loan.any()
    assert inter.is_free.all()


def test_loan_detected_even_if_loan_club_outside_leagues():
    inter, _, _ = run([
        t(13, "2020-08-01", 1, 9, "Alpha FC", "Outside FC", 0),
        t(13, "2021-06-30", 9, 1, "Outside FC", "Alpha FC", 0),
    ])
    # the loan goes to a club outside the leagues and the return is dropped
    assert len(inter) == 0


def test_pseudo_clubs_youth_and_promotions_dropped():
    inter, _, _ = run([
        t(14, "2022-07-01", 1, 0, "Alpha FC", "Retired", 0),
        t(15, "2022-07-01", 1, 50, "Alpha FC", "Alpha FC U19", 0),
        t(16, "2022-07-01", 50, 1, "Alpha FC U19", "Alpha FC", 0),   # promotion
        t(17, "2022-07-01", 60, 1, "Beta FC U19", "Alpha FC", 0),    # bought from another academy
        t(18, "2022-07-01", 515, 3, "Without Club", "Gamma FC", 0),  # free agent signing
    ])
    assert sorted(inter.player_id) == [17, 18]


def test_future_and_duplicates_and_outside_league():
    inter, _, _ = run([
        t(19, "2026-08-01", 2, 1, "Beta FC", "Alpha FC", 1e6),       # after snapshot date
        t(20, "2021-07-01", 2, 1, "Beta FC", "Alpha FC", 1e6),
        t(20, "2021-07-01", 2, 1, "Beta FC", "Alpha FC", 1e6),       # duplicate
        t(21, "2021-07-01", 1, 9, "Alpha FC", "Outside FC", 1e6),    # recruiter outside leagues
    ])
    assert list(inter.player_id) == [20]


def test_min_club_transfers():
    rows = [t(100 + i, "2021-07-01", 2, 1, "Beta FC", "Alpha FC", 1e6) for i in range(3)]
    rows.append(t(200, "2021-07-01", 1, 3, "Alpha FC", "Gamma FC", 1e6))
    inter, clubs, _ = run(rows, min_club=2)
    assert set(inter.club_id) == {1}
    assert list(clubs.club_id) == [1] and clubs.n_transfers.iloc[0] == 3
