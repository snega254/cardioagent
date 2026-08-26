"""
Email-based authentication using bcrypt (properly salted, properly hashed).
Session persistence is via Streamlit's session_state.
"""
import re
import bcrypt


def hash_password(plain_password):
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password, password_hash):
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))


def validate_email(email):
    if not email or len(email) < 3:
        return False, "Email must be at least 3 characters."
    # Basic email validation
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        return False, "Please enter a valid email address."
    return True, ""


def validate_password(password):
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."
    return True, ""


def validate_name(name):
    if not name or len(name) < 2:
        return False, "Name must be at least 2 characters."
    return True, ""