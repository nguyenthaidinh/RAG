"""
RBAC role definitions and validation for the AI Server.

Roles:
  system_admin  – global: create tenants, manage everything
  tenant_admin  – scoped: manage users & keys within own tenant
  user          – scoped: consume APIs only
"""

ROLE_SYSTEM_ADMIN = "system_admin"
ROLE_TENANT_ADMIN = "tenant_admin"
ROLE_USER = "user"

VALID_ROLES = frozenset({ROLE_SYSTEM_ADMIN, ROLE_TENANT_ADMIN, ROLE_USER})

# Roles that count as "admin" (can access admin endpoints)
ADMIN_ROLES = frozenset({ROLE_SYSTEM_ADMIN, ROLE_TENANT_ADMIN})

# Roles a tenant_admin is allowed to assign
TENANT_ADMIN_ASSIGNABLE_ROLES = frozenset({ROLE_TENANT_ADMIN, ROLE_USER})

# Roles a system_admin is allowed to assign
SYSTEM_ADMIN_ASSIGNABLE_ROLES = frozenset(VALID_ROLES)


def is_admin(role: str) -> bool:
    return role in ADMIN_ROLES


def is_system_admin(role: str) -> bool:
    return role == ROLE_SYSTEM_ADMIN


def validate_role(role: str) -> str:
    """Validate and return the role string. Raises ValueError if invalid."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of: {', '.join(sorted(VALID_ROLES))}")
    return role
