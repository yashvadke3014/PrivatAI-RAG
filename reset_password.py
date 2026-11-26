"""Set a new password for an existing user.

    py reset_password.py <username>
"""
import getpass
import sys

from app import auth, db

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

username = sys.argv[1]
password = getpass.getpass("New password (min 8 chars): ")
if len(password) < 8:
    print("Password must be at least 8 characters.")
    sys.exit(1)
if password != getpass.getpass("Repeat password: "):
    print("Passwords don't match.")
    sys.exit(1)

try:
    new_hash = auth.hash_password(password)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

with db.get_conn() as conn:
    changed = conn.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username)
    ).rowcount

print(f"Password updated for '{username}'." if changed else f"No user named '{username}'.")