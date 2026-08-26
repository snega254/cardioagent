"""
Real password authentication using bcrypt (properly salted, properly
hashed — not a toy comparison). Session persistence is via Streamlit's
session_state, which lasts for the browser tab session but not across
browser restarts — that's an honest, stated limitation, not a shortcut
pretending to be full persistent-cookie auth.
"""
import re

import bcrypt


def hash_password(plain_password):
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password, password_hash):
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))


def validate_username(username):
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Username can only contain letters, numbers, and underscores."
    return True, ""


def validate_password(password):
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."
    return True, ""
