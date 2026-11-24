"""Central settings. Anything environment-specific can be overridden with an env var."""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "enterprise_documents"

DB_PATH = BASE_DIR / "privatai.db"
UPLOAD_DIR = BASE_DIR / "uploads"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
# Plain text only for now. PDFs/scans come later with the OCR stretch goal.
ALLOWED_EXTENSIONS = {".txt", ".md"}
TOKEN_EXPIRE_MINUTES = 60

# The only teams that exist. Add more here as needed.
TEAMS = ["Legal", "Finance", "HR"]


def canonical_team(name: str) -> str | None:
    """'finance ' -> 'Finance'. Returns None if the team doesn't exist."""
    for team in TEAMS:
        if team.lower() == name.strip().lower():
            return team
    return None


def get_jwt_secret() -> str:
    """Secret used to sign login tokens.

    Uses the JWT_SECRET env var if set; otherwise generates one on first run
    and saves it to jwt_secret.key (same idea as secret.key in the
    steganography project: persist it so tokens survive a restart, and keep
    it out of git).
    """
    from_env = os.getenv("JWT_SECRET")
    if from_env:
        return from_env
    secret_file = BASE_DIR / "jwt_secret.key"
    if secret_file.exists():
        return secret_file.read_text().strip()
    secret = secrets.token_hex(32)
    secret_file.write_text(secret)
    return secret