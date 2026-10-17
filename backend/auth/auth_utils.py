import re
import logging
import unicodedata
from typing import Optional

import bcrypt
from config import Config

logger = logging.getLogger(__name__)

_MAX_PASSWORD_LENGTH = 128

_DUMMY_HASH = bcrypt.hashpw(b"sentinel_password", bcrypt.gensalt(rounds=4))

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")

_EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)

_USERNAME_PATTERN = re.compile(r"^[a-z0-9_]{3,30}$")


def hash_password(password: str) -> str:

    if not password:
        raise ValueError("Password cannot be empty")

    if len(password) > _MAX_PASSWORD_LENGTH:
        raise ValueError("Password too long")

    salt = bcrypt.gensalt(rounds=Config.BCRYPT_LOG_ROUNDS)

    hashed = bcrypt.hashpw(
        password.encode("utf-8"),
        salt,
    )

    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:

    if not password or not password_hash:
        bcrypt.checkpw(b"dummy", _DUMMY_HASH)
        return False

    try:

        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )

    except Exception:

        bcrypt.checkpw(b"dummy", _DUMMY_HASH)
        return False


def is_password_strong(password: str) -> Optional[str]:

    if not password:
        return "Password must not be empty."

    if len(password) < Config.PASSWORD_MIN_LENGTH:
        return f"Password must be at least {Config.PASSWORD_MIN_LENGTH} characters."

    if len(password) > _MAX_PASSWORD_LENGTH:
        return f"Password must not exceed {_MAX_PASSWORD_LENGTH} characters."

    if Config.PASSWORD_REQUIRE_UPPERCASE and not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."

    if Config.PASSWORD_REQUIRE_DIGIT and not re.search(r"\d", password):
        return "Password must contain at least one digit."

    if Config.PASSWORD_REQUIRE_SPECIAL and not re.search(
        r"[!@#$%^&*(),.?\":{}|<>]", password
    ):
        return "Password must contain at least one special character."

    if len(set(password)) < 3:
        return "Password too simple."

    return None


def sanitize_string(value: str, max_length: int = 255) -> str:

    if not isinstance(value, str):
        value = str(value)

    value = unicodedata.normalize("NFC", value)

    value = _CONTROL_CHAR_RE.sub("", value)

    value = _HTML_TAG_RE.sub("", value)

    value = re.sub(r"(javascript|vbscript):", "", value, flags=re.IGNORECASE)

    value = " ".join(value.split())

    return value[:max_length].strip()


def sanitize_email(email: str) -> str:

    if not isinstance(email, str):
        return ""

    email = unicodedata.normalize("NFC", email)

    email = email.strip().lower()

    email = re.sub(r"[^\w.@+\-]", "", email)

    return email[:255]


def is_valid_email(email: str) -> bool:

    if not email:
        return False

    email = email.strip().lower()

    return bool(_EMAIL_PATTERN.match(email))


def is_valid_username(username: str) -> bool:

    if not username:
        return False

    username = username.strip().lower()

    if not _USERNAME_PATTERN.match(username):
        return False

    if username.startswith("_") or username.endswith("_"):
        return False

    if "__" in username:
        return False

    return True


def mask_email(email: str) -> str:

    if not email or "@" not in email:
        return "***"

    local, domain = email.split("@", 1)

    return f"{local[:2]}***@{domain}"


def mask_username(username: str) -> str:

    if not username:
        return "***"

    return f"{username[:2]}***"