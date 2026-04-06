from datetime import datetime, timedelta, timezone
from jose import jwt

from app.core.config import settings


def create_access_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(seconds=settings.ACCESS_TOKEN_EXPIRE_SECONDS)
    
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALG,
    )
