"""Passwords, login tokens, and the 'who is this user?' dependency."""
import datetime as dt

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from . import config, db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password(password: str) -> bytes:
    raw = password.encode()
    if len(raw) > 72:  # bcrypt only looks at the first 72 bytes
        raise ValueError("Password too long (max 72 bytes)")
    return bcrypt.hashpw(raw, bcrypt.gensalt())


def verify_password(password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed)
    except ValueError:
        return False


def create_access_token(username: str) -> str:
    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=config.TOKEN_EXPIRE_MINUTES)
    # The token only says WHO you are. Your team is looked up from the
    # database on every request, so it can't be forged or go stale.
    return jwt.encode({"sub": username, "exp": expires}, config.get_jwt_secret(), algorithm="HS256")


def create_account(username: str, password: str, team: str, role: str = "user") -> None:
    """Validate and create a user. Raises ValueError (bad input) or
    sqlite3.IntegrityError (username taken)."""
    canonical = config.canonical_team(team)
    if canonical is None:
        raise ValueError(f"Unknown team '{team}'. Valid teams: {', '.join(config.TEAMS)}")
    if role not in ("user", "admin"):
        raise ValueError("Role must be 'user' or 'admin'")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    db.create_user(username, hash_password(password), canonical, role)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    unauthorized = HTTPException(
        status_code=401,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.get_jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        raise unauthorized
    username = payload.get("sub")
    user = db.get_user(username) if username else None
    if user is None or not user["is_active"]:
        raise unauthorized
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user