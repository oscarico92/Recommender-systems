"""Exercise the page against the real API/model and a temporary SQLite database."""

from urllib.parse import urlsplit

import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.interactions import get_db
from api.main import app
from src.config import ROOT_DIR, SERVING_DIR
from src.db import connect

pytestmark = pytest.mark.skipif(
    not (SERVING_DIR / "embeddings.npz").exists(), reason="no exported model"
)


@pytest.fixture
def page_api(monkeypatch, tmp_path):
    db_path = tmp_path / "actions.db"

    def db():
        conn = connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    def as_requests_response(response):
        result = requests.Response()
        result.status_code = response.status_code
        result._content = response.content
        result.url = str(response.url)
        return result

    st.cache_data.clear()
    app.dependency_overrides[get_db] = db
    with TestClient(app) as client:
        monkeypatch.setattr(requests, "get", lambda url, **kw: as_requests_response(
            client.get(urlsplit(url).path, params=kw.get("params"))
        ))
        monkeypatch.setattr(requests, "post", lambda url, **kw: as_requests_response(
            client.post(urlsplit(url).path, json=kw.get("json"))
        ))
        try:
            yield client
        finally:
            app.dependency_overrides.pop(get_db, None)
            st.cache_data.clear()


def open_page(account: dict | None = None):
    """The page of a signed-in scout of Manchester United (985), unless another account is given."""
    page = AppTest.from_file(str(ROOT_DIR / "app/streamlit_app.py"), default_timeout=20)
    page.session_state["account"] = account or {"username": "test_scout", "club_id": 985}
    return page.run()


def open_login_page():
    return AppTest.from_file(str(ROOT_DIR / "app/streamlit_app.py"), default_timeout=20).run()


@pytest.mark.parametrize("label,action", [("Shortlist", "shortlisted"), ("Dismiss", "dismissed")])
def test_action_refreshes_list_and_is_saved_once(page_api, label, action):
    page = open_page()
    assert not page.exception
    button = next(b for b in page.button if b.label == label)
    player_id = int(button.key[len("main") + 1:])
    page = button.click().run()
    assert not page.exception
    rows = page_api.get("/interactions", params={"club_id": 985}).json()
    assert len(rows) == 1
    assert rows[0]["player_id"] == player_id and rows[0]["action"] == action
    assert all(b.key not in {f"mains{player_id}", f"maind{player_id}"} for b in page.button)


@pytest.mark.parametrize("path", ["/config", "/recommend", "/interactions"])
def test_unavailable_endpoint_shows_message_not_traceback(page_api, monkeypatch, path):
    get = requests.get

    def broken(url, **kwargs):
        if urlsplit(url).path == path:
            raise requests.ConnectionError("API unavailable")
        return get(url, **kwargs)

    monkeypatch.setattr(requests, "get", broken)
    page = open_page()
    assert not page.exception
    assert len(page.error) + len(page.warning) > 0
    if path == "/interactions":
        assert not any("No action saved" in item.value for item in page.markdown)
    if path == "/config":
        assert not any(b.label in {"Shortlist", "Dismiss"} for b in page.button)


def test_failed_save_does_not_report_success(page_api, monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("Saving timed out")

    monkeypatch.setattr(requests, "post", fail)
    page = open_page()
    button = next(b for b in page.button if b.label == "Shortlist")
    key = button.key
    page = button.click().run()
    assert not page.exception
    assert page.error and not page.toast
    assert any(b.key == key for b in page.button)
    assert page_api.get("/interactions").json() == []


def test_low_budget_players_are_marked(page_api):
    page = open_page()
    budget = next(s for s in page.slider if s.label == "Budget per player (M EUR)")
    page = budget.set_value(0.5).run()
    assert not page.exception
    assert any("Above budget" in w.value for w in page.warning)


def test_sliders_start_from_served_model(page_api):
    page = open_page()
    cfg = page_api.get("/config").json()
    values = {s.label: s.value for s in page.slider}
    assert values["Resale focus"] == cfg["resale_weight"]
    assert values["Youth focus"] == cfg["age_weight"]
    assert values["Penalty for frequent movers"] == cfg["mobility_weight"]


def test_non_json_api_error_is_handled(page_api, monkeypatch):
    get = requests.get

    def unavailable(url, **kwargs):
        if urlsplit(url).path == "/recommend":
            response = requests.Response()
            response.status_code = 503
            response._content = b"<html>Service unavailable</html>"
            return response
        return get(url, **kwargs)

    monkeypatch.setattr(requests, "get", unavailable)
    page = open_page()
    assert not page.exception and page.error


def test_shortlist_tab_lists_and_compares_shortlisted_players(page_api):
    page = open_page()
    for _ in range(2):
        button = next(b for b in page.button if b.label == "Shortlist" and b.key.startswith("main"))
        page = button.click().run()
        assert not page.exception
    assert any(s.value == "Your shortlist (2)" for s in page.subheader)
    picker = next(m for m in page.multiselect if m.label.startswith("Choose 2 or 3 players"))
    assert len(picker.value) == 2


def test_table_view_renders(page_api):
    page = open_page()
    page = page.radio[0].set_value("Table").run()
    assert not page.exception
    assert page.dataframe


def test_twenty_recommendations_keep_new_talent_profiles(page_api):
    page = open_page()
    page = next(s for s in page.slider if s.label == "Number of players").set_value(20).run()
    assert not page.exception
    for toggle in [t for t in page.toggle if t.label == "Scouting report"]:
        toggle.set_value(True)
    page = page.run()
    assert not page.exception
    assert not any(c.value == "No detailed statistics for this player." for c in page.caption)


def test_scouting_reports_are_drawn_only_when_opened(page_api):
    """Drawing every report on every rerun (33 charts) made each click slow in the browser."""
    def reports(page):
        return [c for c in page.get("plotly_chart") if c.key != "compare_radar"]

    page = open_page()
    assert reports(page) == []
    toggle = next(t for t in page.toggle if t.label == "Scouting report")
    page = toggle.set_value(True).run()
    assert not page.exception
    charts = reports(page)
    assert len(charts) == 2                       # radar + value curve of that player
    keys = {c.key for c in charts}
    page = page.run()                             # a rerun keeps the same chart keys
    assert {c.key for c in reports(page)} == keys


def test_shortlist_over_twenty_players_keeps_every_profile(page_api):
    import pandas as pd

    ids = pd.read_parquet(SERVING_DIR / "player_profiles.parquet")["player_id"].head(25).tolist()
    for pid in ids:
        page_api.post("/log_interaction", json={"club_id": 985, "player_id": int(pid), "action": "shortlisted"})
    page = open_page()
    assert not page.exception
    assert any(s.value == "Your shortlist (25)" for s in page.subheader)
    assert not any(f"Player {pid}<" in m.value for m in page.markdown for pid in ids)
    picker = next(m for m in page.multiselect if m.label.startswith("Choose 2 or 3 players"))
    assert {page_api.get("/players", params={"ids": str(pid)}).json()[0]["name"] for pid in ids} <= {
        o.split(" (")[0] for o in picker.options
    }


def test_login_page_is_shown_first(page_api):
    page = open_login_page()
    assert not page.exception
    assert [t.label for t in page.tabs] == ["Sign in", "Create an account"]
    assert not any(b.label in {"Shortlist", "Dismiss"} for b in page.button)


def fill_sign_up(page, user, pwd, pwd2, club_prefix):
    page.text_input(key="up_user").input(user)
    page.text_input(key="up_pwd").input(pwd)
    page.text_input(key="up_pwd2").input(pwd2)
    box = page.selectbox(key="up_club")
    box.set_value(next(o for o in box.options if o.startswith(club_prefix)))
    return page


def test_create_account_then_sign_in_with_the_chosen_club(page_api):
    page = fill_sign_up(open_login_page(), "scout_celtic", "secret123", "secret123", "Celtic")
    page = page.button[1].click().run()          # "Create account" (second form button)
    assert not page.exception
    assert page.session_state["account"]["username"] == "scout_celtic"
    assert "Celtic" in " ".join(m.value for m in page.markdown)
    assert any(b.label == "Shortlist" for b in page.button)

    # log out, then sign in again with the same account
    page = next(b for b in page.button if b.label == "Log out").click().run()
    assert [t.label for t in page.tabs] == ["Sign in", "Create an account"]
    page.text_input(key="in_user").input("scout_celtic")
    page.text_input(key="in_pwd").input("secret123")
    page = page.button[0].click().run()
    assert not page.exception
    assert page.session_state["account"]["club_id"] == 371


def test_wrong_password_and_duplicate_username_are_refused(page_api):
    page_api.post("/accounts", json={"username": "scout_benfica", "password": "secret123", "club_id": 294})
    page = open_login_page()
    page.text_input(key="in_user").input("scout_benfica")
    page.text_input(key="in_pwd").input("wrong-password")
    page = page.button[0].click().run()
    assert "account" not in page.session_state
    assert any("Wrong username or password" in e.value for e in page.error)

    page = fill_sign_up(open_login_page(), "Scout_Benfica", "secret123", "secret123", "Manchester United")
    page = page.button[1].click().run()
    assert "account" not in page.session_state
    assert any("already taken" in e.value for e in page.error)


def test_different_passwords_are_refused(page_api):
    page = fill_sign_up(open_login_page(), "scout_x", "secret123", "secret124", "Manchester United")
    page = page.button[1].click().run()
    assert "account" not in page.session_state
    assert any("different" in e.value for e in page.error)
