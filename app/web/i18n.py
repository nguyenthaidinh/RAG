"""
Internationalization (i18n) — simple cookie-based translation system.

Usage in templates::

    {{ t('dashboard') }}         → "Dashboard" or "Bảng điều khiển"
    {{ t('save_changes') }}      → "Save Changes" or "Lưu thay đổi"

Language is stored in a cookie ``lang`` (default: ``vi``).
"""
from __future__ import annotations

from typing import Any

# ── Translation dictionary ────────────────────────────────────────────
# Key → { "en": "...", "vi": "..." }

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── Layout / Navigation ──
    "admin_console": {"en": "Admin Console", "vi": "Bảng quản trị"},
    "dashboard": {"en": "Dashboard", "vi": "Bảng điều khiển"},
    "tenants": {"en": "Tenants", "vi": "Khách hàng"},
    "management": {"en": "Management", "vi": "Quản lý"},
    "users": {"en": "Users", "vi": "Người dùng"},
    "api_keys": {"en": "API Keys", "vi": "Khóa API"},
    "documents": {"en": "Documents", "vi": "Tài liệu"},
    "usage": {"en": "Usage", "vi": "Sử dụng"},
    "plan_quotas": {"en": "Plan & Quotas", "vi": "Gói & Hạn mức"},
    "monitoring": {"en": "Monitoring", "vi": "Giám sát"},
    "audit_log": {"en": "Audit Log", "vi": "Nhật ký kiểm tra"},
    "ops_dashboard": {"en": "Ops Dashboard", "vi": "Vận hành"},
    "tools": {"en": "Tools", "vi": "Công cụ"},
    "api_test": {"en": "API Test", "vi": "Kiểm tra API"},
    "logout": {"en": "Logout", "vi": "Đăng xuất"},

    # ── Common actions ──
    "save_changes": {"en": "Save Changes", "vi": "Lưu thay đổi"},
    "create": {"en": "Create", "vi": "Tạo mới"},
    "edit": {"en": "Edit", "vi": "Chỉnh sửa"},
    "delete": {"en": "Delete", "vi": "Xóa"},
    "cancel": {"en": "Cancel", "vi": "Hủy"},
    "search": {"en": "Search", "vi": "Tìm kiếm"},
    "reset": {"en": "Reset", "vi": "Đặt lại"},
    "view": {"en": "View", "vi": "Xem"},
    "back": {"en": "Back", "vi": "Quay lại"},
    "loading": {"en": "Loading…", "vi": "Đang tải…"},
    "saving": {"en": "Saving…", "vi": "Đang lưu…"},
    "saved": {"en": "Saved!", "vi": "Đã lưu!"},
    "creating": {"en": "Creating…", "vi": "Đang tạo…"},
    "actions": {"en": "Actions", "vi": "Thao tác"},
    "status": {"en": "Status", "vi": "Trạng thái"},
    "active": {"en": "Active", "vi": "Hoạt động"},
    "inactive": {"en": "Không hoạt động", "vi": "Không hoạt động"},
    "enabled": {"en": "Enabled", "vi": "Bật"},
    "disabled": {"en": "Disabled", "vi": "Tắt"},
    "all": {"en": "All", "vi": "Tất cả"},
    "none": {"en": "None", "vi": "Không có"},

    # ── Auth / Login ──
    "login": {"en": "Login", "vi": "Đăng nhập"},
    "login_title": {"en": "Login to system", "vi": "Đăng nhập hệ thống"},
    "email": {"en": "Email", "vi": "Email"},
    "password": {"en": "Password", "vi": "Mật khẩu"},
    "remember_me": {"en": "Remember me", "vi": "Ghi nhớ đăng nhập"},
    "login_button": {"en": "Login", "vi": "Đăng nhập"},
    "login_error": {"en": "Invalid email or password", "vi": "Email hoặc mật khẩu không đúng"},
    "show": {"en": "Show", "vi": "Hiện"},
    "hide": {"en": "Hide", "vi": "Ẩn"},

    # ── Dashboard ──
    "welcome": {"en": "Welcome", "vi": "Xin chào"},
    "role": {"en": "Role", "vi": "Vai trò"},
    "tenant": {"en": "Tenant", "vi": "Tenant"},
    "query_analytics": {"en": "Query Analytics", "vi": "Phân tích truy vấn"},
    "queries_24h": {"en": "Queries (24h)", "vi": "Truy vấn (24h)"},
    "queries_7d": {"en": "Queries (7d)", "vi": "Truy vấn (7 ngày)"},
    "tokens_used_7d": {"en": "Tokens Used (7d)", "vi": "Token đã dùng (7 ngày)"},
    "avg_latency_24h": {"en": "Avg Latency (24h)", "vi": "Độ trễ TB (24h)"},
    "top_tenant_7d": {"en": "Top Tenant (7d)", "vi": "Tenant hàng đầu (7 ngày)"},
    "quick_actions": {"en": "Quick Actions", "vi": "Thao tác nhanh"},
    "manage_users": {"en": "Manage Users", "vi": "Quản lý người dùng"},
    "manage_users_desc": {"en": "View and manage user accounts", "vi": "Xem và quản lý tài khoản người dùng"},
    "api_keys_desc": {"en": "View and manage API keys", "vi": "Xem và quản lý khóa API"},
    "api_test_desc": {"en": "Test API endpoints directly", "vi": "Kiểm tra API trực tiếp"},
    "operations": {"en": "Operations", "vi": "Vận hành"},
    "operations_desc": {"en": "Real-time system monitoring", "vi": "Giám sát hệ thống thời gian thực"},
    "authentication": {"en": "Authentication", "vi": "Xác thực"},
    "secure": {"en": "Secure", "vi": "An toàn"},
    "auth_desc": {"en": "HttpOnly cookie · Tokens never exposed to browser", "vi": "Cookie HttpOnly · Token không bao giờ lộ ra trình duyệt"},
    "runtime": {"en": "Runtime", "vi": "Nền tảng"},

    # ── User Management ──
    "users_management": {"en": "Users Management", "vi": "Quản lý người dùng"},
    "new_user": {"en": "New User", "vi": "Thêm người dùng"},
    "create_user": {"en": "Create User", "vi": "Tạo người dùng"},
    "search_email": {"en": "Search email...", "vi": "Tìm theo email..."},
    "all_status": {"en": "All Status", "vi": "Tất cả trạng thái"},
    "no_users_found": {"en": "No users found", "vi": "Không tìm thấy người dùng"},
    "user_detail": {"en": "User Detail", "vi": "Chi tiết người dùng"},
    "tenant_id": {"en": "Tenant ID", "vi": "Mã khách hàng"},
    "email_required": {"en": "Email is required", "vi": "Email là bắt buộc"},
    "password_required": {"en": "Password is required", "vi": "Mật khẩu là bắt buộc"},
    "tenant_id_required": {"en": "Tenant ID is required", "vi": "Mã khách hàng là bắt buộc"},
    "created": {"en": "Created", "vi": "Ngày tạo"},
    "previous": {"en": "Previous", "vi": "Trước"},
    "next": {"en": "Next", "vi": "Sau"},

    # ── Tenant Management ──
    "tenant_management": {"en": "Tenant Management", "vi": "Quản lý khách hàng"},
    "total_tenants": {"en": "Total Tenants", "vi": "Tổng khách hàng"},
    "all_tenants": {"en": "All Tenants", "vi": "Tất cả khách hàng"},
    "new_tenant": {"en": "New Tenant", "vi": "Thêm khách hàng"},
    "create_tenant": {"en": "Create Tenant", "vi": "Tạo khách hàng"},
    "tenant_name": {"en": "Tenant Name", "vi": "Tên khách hàng"},
    "max_users": {"en": "Max Users", "vi": "Số người dùng tối đa"},
    "max_requests": {"en": "Max Requests", "vi": "Số yêu cầu tối đa"},
    "max_tokens": {"en": "Max Tokens", "vi": "Số token tối đa"},
    "max_storage": {"en": "Max Storage (MB)", "vi": "Dung lượng tối đa (MB)"},
    "tenant_id_help": {"en": "Unique identifier. Lowercase, 3-64 chars. Cannot be changed after creation.",
                       "vi": "Mã định danh duy nhất. Chữ thường, 3-64 ký tự. Không đổi được sau khi tạo."},
    "tenant_name_help": {"en": "Display name for this tenant.", "vi": "Tên hiển thị của khách hàng."},
    "monthly_request_quota": {"en": "Monthly request quota.", "vi": "Hạn mức yêu cầu hàng tháng."},
    "monthly_token_quota": {"en": "Monthly token quota.", "vi": "Hạn mức token hàng tháng."},
    "doc_storage_limit": {"en": "Document storage limit.", "vi": "Giới hạn dung lượng tài liệu."},
    "no_tenants_yet": {"en": "No tenants yet", "vi": "Chưa có khách hàng"},
    "no_tenants_desc": {"en": "Create your first tenant to get started with multi-tenant management.",
                        "vi": "Tạo khách hàng đầu tiên để bắt đầu quản lý đa khách hàng."},
    "tenant_detail": {"en": "Tenant Detail", "vi": "Chi tiết khách hàng"},
    "edit_tenant": {"en": "Edit Tenant", "vi": "Chỉnh sửa khách hàng"},
    "users_in_tenant": {"en": "Users in Tenant", "vi": "Người dùng trong tenant"},
    "no_users_yet": {"en": "No users yet", "vi": "Chưa có người dùng"},
    "no_users_in_tenant": {"en": "No users belong to this tenant.", "vi": "Không có người dùng trong khách hàng này."},
    "users_count": {"en": "Users Count", "vi": "Số người dùng"},
    "name": {"en": "Name", "vi": "Tên"},

    "tenant_updated": {"en": "Tenant updated successfully", "vi": "Cập nhật khách hàng thành công"},
    "tenant_update_failed": {"en": "Failed to update tenant", "vi": "Cập nhật khách hàng thất bại"},
    "network_error": {"en": "Network error", "vi": "Lỗi mạng"},

    # ── Settings ──
    "settings": {"en": "Settings", "vi": "Cài đặt"},
    "current_plan": {"en": "Current Plan", "vi": "Gói hiện tại"},
    "plan_code": {"en": "Plan Code", "vi": "Mã gói"},
    "rate_limit": {"en": "Rate Limit", "vi": "Giới hạn tốc độ"},
    "daily_token_quota": {"en": "Daily Token Quota", "vi": "Hạn mức token/ngày"},
    "monthly_token_quota": {"en": "Monthly Token Quota", "vi": "Hạn mức token/tháng"},
    "user_rate_limit": {"en": "User Rate Limit", "vi": "Giới hạn người dùng"},
    "update_settings": {"en": "Update Settings", "vi": "Cập nhật cài đặt"},
    "unlimited": {"en": "Unlimited", "vi": "Không giới hạn"},
    "save_settings": {"en": "Save Settings", "vi": "Lưu cài đặt"},

    # ── Ops ──
    "system_status": {"en": "System Status", "vi": "Trạng thái hệ thống"},
    "uptime": {"en": "Uptime", "vi": "Thời gian hoạt động"},
    "in_flight": {"en": "In-Flight", "vi": "Đang xử lý"},
    "error_rate": {"en": "Error Rate", "vi": "Tỷ lệ lỗi"},
    "request_counts": {"en": "Request Counts", "vi": "Số lượng yêu cầu"},
    "latency": {"en": "Latency", "vi": "Độ trễ"},
    "backpressure": {"en": "Backpressure", "vi": "Áp lực ngược"},
    "db_pool": {"en": "DB Pool", "vi": "Kết nối DB"},
    "top_tenants": {"en": "Top Tenants (by queries)", "vi": "Khách hàng hàng đầu (theo truy vấn)"},

    # ── Language ──
    "language": {"en": "Language", "vi": "Ngôn ngữ"},
    "english": {"en": "English", "vi": "Tiếng Anh"},
    "vietnamese": {"en": "Vietnamese", "vi": "Tiếng Việt"},
}

# Default language
DEFAULT_LANG = "vi"
SUPPORTED_LANGS = ("en", "vi")


def get_translator(lang: str):
    """
    Return a translation function bound to `lang`.

    Usage::

        t = get_translator("vi")
        t("dashboard")  # → "Bảng điều khiển"
    """
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG

    def t(key: str, **kwargs: Any) -> str:
        entry = TRANSLATIONS.get(key)
        if not entry:
            return key  # fallback: return key itself

        text = entry.get(lang, entry.get("en", key))

        # Simple string formatting if kwargs provided
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text

        return text

    return t
