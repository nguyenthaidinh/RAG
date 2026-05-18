import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1 import admin_api_keys


class FakeDB:
    def __init__(self):
        self.committed = False
        self.rolled_back = False

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


def _actor(*, role: str, tenant_id: str = "tenant_a", user_id: int = 1):
    return SimpleNamespace(id=user_id, role=role, tenant_id=tenant_id)


def _user(*, user_id: int = 10, tenant_id: str = "tenant_a"):
    return SimpleNamespace(id=user_id, tenant_id=tenant_id)


def _api_key(
    *,
    key_id: int = 100,
    user_id: int = 10,
    tenant_id: str = "tenant_a",
    name: str = "Client key",
    is_active: bool = True,
    revoked_at=None,
):
    return SimpleNamespace(
        id=key_id,
        user_id=user_id,
        tenant_id=tenant_id,
        name=name,
        prefix="ak_live_1234567890abcdef",
        secret_hash="must_not_leak",
        is_active=is_active,
        revoked_at=revoked_at,
        last_used_at=None,
        created_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )


def _json_response(response):
    return response.model_dump(mode="json")


def _assert_no_secret_hash(payload):
    assert "secret_hash" not in json.dumps(payload, default=str)
    assert "must_not_leak" not in json.dumps(payload, default=str)


@pytest.mark.asyncio
async def test_system_admin_create_key_for_user_returns_plain_key_without_hash():
    db = FakeDB()
    key = _api_key(tenant_id="tenant_b")

    with (
        patch.object(
            admin_api_keys.UserService,
            "get_by_id",
            AsyncMock(return_value=_user(tenant_id="tenant_b")),
        ),
        patch.object(
            admin_api_keys.APIKeyService,
            "create_api_key",
            AsyncMock(return_value=(key, "ak_live_1234567890abcdef.secret")),
        ) as create_mock,
    ):
        response = await admin_api_keys.create_api_key(
            admin_api_keys.CreateAPIKeyRequest(user_id=10, name="Client key"),
            db=db,
            admin=_actor(role="system_admin", tenant_id="root"),
        )

    payload = _json_response(response)
    assert payload["ok"] is True
    assert payload["plain_key"] == "ak_live_1234567890abcdef.secret"
    assert payload["api_key"]["tenant_id"] == "tenant_b"
    assert db.committed is True
    create_mock.assert_awaited_once()
    _assert_no_secret_hash(payload)


@pytest.mark.asyncio
async def test_tenant_admin_create_key_for_same_tenant_returns_ok():
    db = FakeDB()
    key = _api_key(tenant_id="tenant_a")

    with (
        patch.object(
            admin_api_keys.UserService,
            "get_by_id",
            AsyncMock(return_value=_user(tenant_id="tenant_a")),
        ),
        patch.object(
            admin_api_keys.APIKeyService,
            "create_api_key",
            AsyncMock(return_value=(key, "ak_live_1234567890abcdef.secret")),
        ),
    ):
        response = await admin_api_keys.create_api_key(
            admin_api_keys.CreateAPIKeyRequest(user_id=10, name="Client key"),
            db=db,
            admin=_actor(role="tenant_admin", tenant_id="tenant_a"),
        )

    payload = _json_response(response)
    assert payload["ok"] is True
    assert payload["api_key"]["tenant_id"] == "tenant_a"
    _assert_no_secret_hash(payload)


@pytest.mark.asyncio
async def test_tenant_admin_create_key_for_other_tenant_is_hidden():
    db = FakeDB()

    with (
        patch.object(
            admin_api_keys.UserService,
            "get_by_id",
            AsyncMock(return_value=_user(tenant_id="tenant_b")),
        ),
        patch.object(
            admin_api_keys.APIKeyService,
            "create_api_key",
            AsyncMock(),
        ) as create_mock,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await admin_api_keys.create_api_key(
                admin_api_keys.CreateAPIKeyRequest(user_id=10, name="Client key"),
                db=db,
                admin=_actor(role="tenant_admin", tenant_id="tenant_a"),
            )

    assert exc_info.value.status_code == 404
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoke_key_returns_inactive_key_without_hash():
    db = FakeDB()
    key = _api_key()
    revoked_at = datetime(2026, 5, 18, 1, 0, tzinfo=timezone.utc)

    async def revoke_side_effect(*args, **kwargs):
        key.is_active = False
        key.revoked_at = revoked_at
        return key

    with (
        patch.object(
            admin_api_keys.APIKeyService,
            "get_by_id",
            AsyncMock(return_value=key),
        ),
        patch.object(
            admin_api_keys.APIKeyService,
            "revoke_api_key",
            AsyncMock(side_effect=revoke_side_effect),
        ),
    ):
        response = await admin_api_keys.revoke_api_key(
            key_id=100,
            db=db,
            admin=_actor(role="tenant_admin", tenant_id="tenant_a"),
        )

    payload = _json_response(response)
    assert payload["ok"] is True
    assert payload["api_key"]["is_active"] is False
    assert payload["api_key"]["revoked_at"] is not None
    assert db.committed is True
    _assert_no_secret_hash(payload)


@pytest.mark.asyncio
async def test_rotate_key_revokes_old_and_returns_new_plain_key_without_hash():
    db = FakeDB()
    old_key = _api_key(key_id=100, name="Client key")
    new_key = _api_key(key_id=101, name="Client key (rotated)")
    new_key.prefix = "ak_live_fedcba0987654321"
    revoked_at = datetime(2026, 5, 18, 1, 0, tzinfo=timezone.utc)

    async def rotate_side_effect(*args, **kwargs):
        old_key.is_active = False
        old_key.revoked_at = revoked_at
        return new_key, "ak_live_fedcba0987654321.newsecret"

    with (
        patch.object(
            admin_api_keys.APIKeyService,
            "get_by_id",
            AsyncMock(return_value=old_key),
        ),
        patch.object(
            admin_api_keys.APIKeyService,
            "rotate_api_key",
            AsyncMock(side_effect=rotate_side_effect),
        ),
    ):
        response = await admin_api_keys.rotate_api_key(
            key_id=100,
            db=db,
            admin=_actor(role="tenant_admin", tenant_id="tenant_a"),
        )

    payload = _json_response(response)
    assert payload["ok"] is True
    assert payload["plain_key"] == "ak_live_fedcba0987654321.newsecret"
    assert payload["old_key"]["is_active"] is False
    assert payload["old_key"]["revoked_at"] is not None
    assert payload["new_key"]["id"] == 101
    assert db.committed is True
    _assert_no_secret_hash(payload)
