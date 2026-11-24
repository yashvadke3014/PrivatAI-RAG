"""Create a user from the command line (use this to make your first admin).

    py create_user.py <username> <team> [admin|user]

The password is typed at a prompt so it doesn't end up in your shell history.
"""
import getpass
import sqlite3
import sys

from app import auth, db


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    username, team = sys.argv[1], sys.argv[2]
    role = sys.argv[3] if len(sys.argv) > 3 else "user"

    password = getpass.getpass("Password (min 8 chars): ")
    if password != getpass.getpass("Repeat password: "):
        print("Passwords don't match.")
        sys.exit(1)

    db.init_db()
    try:
        auth.create_account(username, password, team, role)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except sqlite3.IntegrityError:
        print(f"Error: user '{username}' already exists.")
        sys.exit(1)
    print(f"Created {role} '{username}' in team '{team}'.")


if __name__ == "__main__":
    main()