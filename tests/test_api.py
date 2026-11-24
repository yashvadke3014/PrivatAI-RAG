"""API-level tests. The RAG engine is replaced with fakes (no Qdrant/Ollama needed),
so these verify login, tokens, roles, and above all that the TEAM always comes
from the user's account."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import auth, config, rag

    monkeypatch.setenv("JWT_SECRET", "test-secret-test-secret-test-secret-123")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")

    calls = {"ask": [], "ingest": []}

    def fake_ask(team, question):
        calls["ask"].append(team)
        return {"answer": f"fake answer for {team}", "sources": []}

    def fake_ingest(path, team, uploaded_by, display_name):
        calls["ingest"].append((team, display_name))

    monkeypatch.setattr(rag, "ask", fake_ask)
    monkeypatch.setattr(rag, "ingest_file", fake_ingest)

    from app.main import app

    with TestClient(app) as c:
        auth.create_account("admin", "adminpass1", "Legal", "admin")
        auth.create_account("lena", "legalpass1", "Legal")
        auth.create_account("fred", "financepass1", "Finance")
        c.calls = calls
        yield c


def login(client, username, password):
    r = client.post("/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def upload(client, headers, name="a.txt", content=b"hello world"):
    return client.post("/documents", headers=headers, files={"file": (name, content, "text/plain")})


def test_wrong_password_rejected(client):
    r = client.post("/login", data={"username": "lena", "password": "nope-nope"})
    assert r.status_code == 401


def test_unknown_user_gets_same_error_as_wrong_password(client):
    a = client.post("/login", data={"username": "ghost", "password": "whatever1"})
    b = client.post("/login", data={"username": "lena", "password": "whatever1"})
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()


def test_endpoints_require_a_token(client):
    assert client.post("/ask", json={"question": "hello?"}).status_code == 401
    assert client.get("/documents").status_code == 401
    assert client.get("/me").status_code == 401


def test_tampered_token_rejected(client):
    token = login(client, "lena", "legalpass1")["Authorization"].split()[1]
    bad = {"Authorization": f"Bearer {token[:-3]}abc"}
    assert client.get("/me", headers=bad).status_code == 401


def test_me_returns_team_from_account(client):
    r = client.get("/me", headers=login(client, "fred", "financepass1"))
    assert r.json() == {"username": "fred", "team": "Finance", "role": "user"}


def test_ask_uses_team_from_account_not_request_body(client):
    """A Legal user claiming to be Finance in the body must still be searched as Legal."""
    h = login(client, "lena", "legalpass1")
    r = client.post("/ask", headers=h, json={"question": "revenue?", "team": "Finance"})
    assert r.status_code == 200
    assert client.calls["ask"] == ["Legal"]


def test_upload_is_tagged_with_uploaders_team(client):
    r = upload(client, login(client, "fred", "financepass1"), "q3.txt")
    assert r.status_code == 201
    assert client.calls["ingest"] == [("Finance", "q3.txt")]


def test_document_lists_are_team_scoped(client):
    legal, finance = login(client, "lena", "legalpass1"), login(client, "fred", "financepass1")
    upload(client, legal, "contract.txt", b"legal content")
    upload(client, finance, "budget.txt", b"finance content")
    assert [d["filename"] for d in client.get("/documents", headers=legal).json()] == ["contract.txt"]
    assert [d["filename"] for d in client.get("/documents", headers=finance).json()] == ["budget.txt"]


def test_same_file_allowed_in_two_teams_but_not_twice_in_one(client):
    legal, finance = login(client, "lena", "legalpass1"), login(client, "fred", "financepass1")
    assert upload(client, legal, "same.txt", b"identical").status_code == 201
    # Another team uploading the same bytes must NOT be told "already uploaded":
    # that would leak the fact that Legal has this file.
    assert upload(client, finance, "same.txt", b"identical").status_code == 201
    assert upload(client, legal, "same.txt", b"identical").status_code == 409


def test_bad_uploads_rejected(client):
    h = login(client, "lena", "legalpass1")
    assert upload(client, h, "virus.exe", b"x").status_code == 400
    assert upload(client, h, "empty.txt", b"   ").status_code == 400
    assert upload(client, h, "binary.txt", b"\xff\xfe\x00\x00").status_code == 400
    assert upload(client, h, "big.txt", b"a" * (5 * 1024 * 1024 + 1)).status_code == 413


def test_path_tricks_in_filename_are_neutralised(client):
    r = upload(client, login(client, "lena", "legalpass1"), "../../evil.txt")
    assert r.status_code == 201
    assert client.calls["ingest"][-1] == ("Legal", "evil.txt")


def test_only_admins_can_create_users(client):
    new = {"username": "newbie", "password": "newbiepass1", "team": "hr"}
    assert client.post("/admin/users", json=new, headers=login(client, "lena", "legalpass1")).status_code == 403
    r = client.post("/admin/users", json=new, headers=login(client, "admin", "adminpass1"))
    assert r.status_code == 201 and r.json()["team"] == "HR"
    assert client.get("/me", headers=login(client, "newbie", "newbiepass1")).json()["team"] == "HR"


def test_admin_cannot_create_user_in_unknown_team(client):
    r = client.post(
        "/admin/users",
        json={"username": "x_user", "password": "password123", "team": "Marketing"},
        headers=login(client, "admin", "adminpass1"),
    )
    assert r.status_code == 400