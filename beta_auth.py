"""
Beta Authentication Service for Frost News Collector

Simple email-based token auth for beta testing.
Tokens are stored in memory and cleared on server restart.
"""

import json
import uuid
from datetime import datetime
from typing import Optional, Dict
import os

from config import BETA_USERS_PATH


# In-memory token store: token -> user_id
_active_tokens: Dict[str, str] = {}


def _load_users() -> list:
    """Load users from JSON file."""
    if not os.path.exists(BETA_USERS_PATH):
        return []
    try:
        with open(BETA_USERS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('users', [])
    except Exception:
        return []


def _save_users(users: list) -> None:
    """Save users to JSON file."""
    with open(BETA_USERS_PATH, 'w', encoding='utf-8') as f:
        json.dump({'users': users}, f, indent=2, ensure_ascii=False)


def get_user_by_email(email: str) -> Optional[dict]:
    """
    Look up a user by email address.
    Returns user dict if found and active, None otherwise.
    """
    email_lower = email.lower().strip()
    users = _load_users()
    for user in users:
        if user.get('email', '').lower() == email_lower:
            if user.get('is_active', False):
                return user
            return None  # User exists but inactive
    return None


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Look up a user by user_id."""
    users = _load_users()
    for user in users:
        if user.get('user_id') == user_id:
            return user
    return None


def generate_token(user_id: str) -> str:
    """
    Generate a new session token for a user.
    Invalidates any existing tokens for this user.
    """
    # Remove any existing tokens for this user
    tokens_to_remove = [t for t, uid in _active_tokens.items() if uid == user_id]
    for t in tokens_to_remove:
        del _active_tokens[t]
    
    # Generate new token
    token = str(uuid.uuid4())
    _active_tokens[token] = user_id
    return token


def validate_token(token: str) -> Optional[dict]:
    """
    Validate a token and return the associated user if valid.
    Returns None if token is invalid or user not found.
    """
    if not token:
        return None
    
    user_id = _active_tokens.get(token)
    if not user_id:
        return None
    
    user = get_user_by_id(user_id)
    if not user or not user.get('is_active', False):
        # Clean up token for inactive user
        del _active_tokens[token]
        return None
    
    return user


def invalidate_token(token: str) -> bool:
    """Invalidate a token (logout). Returns True if token existed."""
    if token in _active_tokens:
        del _active_tokens[token]
        return True
    return False


def update_last_login(user_id: str) -> None:
    """Update the last_login_at timestamp for a user."""
    users = _load_users()
    for user in users:
        if user.get('user_id') == user_id:
            user['last_login_at'] = datetime.utcnow().isoformat() + 'Z'
            break
    _save_users(users)


def can_review(user: dict) -> bool:
    """Check if a user has permission to review/correct labels."""
    role = user.get('role', '')
    return role in ('admin', 'reviewer')


def is_admin(user: dict) -> bool:
    """Check if a user is an admin."""
    return user.get('role') == 'admin'
