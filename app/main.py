"""PrivatAI-RAG API.  Run with:  uvicorn app.main:app --reload"""
import hashlib
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from . import auth, config, db, rag

logger = logging.getLogger("privatai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    config.UPLOAD_DIR.mkdir(exist_ok=True)
    yield


app = FastAPI(title="PrivatAI-RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class NewUser(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=72)
    team: str
    role: Literal["user", "admin"] = "user"


@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = db.get_user(form.username)
    # Same message for "no such user" and "wrong password" so attackers
    # can't discover which usernames exist.
    if (
        user is None
        or not user["is_active"]
        or not auth.verify_password(form.password, user["password_hash"])
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": auth.create_access_token(user["username"]), "token_type": "bearer"}


@app.get("/me")
def me(user: dict = Depends(auth.get_current_user)):
    return {"username": user["username"], "team": user["team"], "role": user["role"]}


@app.post("/admin/users", status_code=201)
def create_user(body: NewUser, _admin: dict = Depends(auth.require_admin)):
    try:
        auth.create_account(body.username, body.password, body.team, body.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="That username is already taken")
    return {"username": body.username, "team": config.canonical_team(body.team), "role": body.role}


# NOTE: these are plain `def` (not `async def`) on purpose. They do slow,
# blocking work (embedding, calling the LLM), and FastAPI runs plain `def`
# endpoints in a thread pool so one slow request doesn't freeze the server.
@app.post("/documents", status_code=201)
def upload_document(file: UploadFile = File(...), user: dict = Depends(auth.get_current_user)):
    display_name = Path(file.filename or "").name  # strip any folder parts
    suffix = Path(display_name).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Only {allowed} files are supported right now")

    content = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB)")
    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text")

    team = user["team"]  # from the account, never from the request
    digest = hashlib.sha256(content).hexdigest()
    try:
        doc_id = db.add_document(team, display_name, digest, user["username"])
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Your team has already uploaded this exact file")

    # Save under a random name: the user-supplied filename is never used as a
    # path, so it can't be abused (e.g. "../../something").
    team_dir = config.UPLOAD_DIR / team
    team_dir.mkdir(parents=True, exist_ok=True)
    saved = team_dir / f"{uuid.uuid4().hex}{suffix}"
    saved.write_bytes(content)
    try:
        rag.ingest_file(saved, team, user["username"], display_name)
    except Exception:
        logger.exception("Indexing failed for %s", display_name)
        db.delete_document(doc_id)
        saved.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not index the document (are Qdrant and Ollama running?)")
    return {"filename": display_name, "team": team}


@app.get("/documents")
def list_my_documents(user: dict = Depends(auth.get_current_user)):
    return db.list_documents(user["team"])


@app.post("/ask")
def ask(body: AskRequest, user: dict = Depends(auth.get_current_user)):
    try:
        return rag.ask(team=user["team"], question=body.question)
    except Exception:
        logger.exception("Question failed")
        raise HTTPException(status_code=500, detail="Could not answer right now (are Qdrant and Ollama running?)")