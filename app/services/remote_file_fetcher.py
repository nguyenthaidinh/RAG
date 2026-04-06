"""
Remote file fetcher — downloads files from temporary URLs (e.g. File-Service).

Uses httpx (already in requirements) for async HTTP.
Provides structured errors for the router layer.

Phase 1.1 hardening:
  - Host allowlist validation via REMOTE_FETCH_ALLOWED_HOSTS.
  - Manual redirect following with per-hop host validation.
  - SSRF guards: blocks localhost, 127.x, ::1, link-local, metadata IPs.
  - Max redirect hops from REMOTE_FETCH_MAX_REDIRECTS.
"""
from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from urllib.parse import urlparse, unquote

import httpx

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_MAX_REDIRECTS = 3


# ── Exceptions ────────────────────────────────────────────────────────

class RemoteFetchError(Exception):
    """Base exception for remote fetch failures."""

    def __init__(self, message: str, *, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class RemoteFetchTimeout(RemoteFetchError):
    """Download timed out."""

    def __init__(self, url_host: str, timeout: float):
        super().__init__(
            f"Timeout downloading file from {url_host} (timeout={timeout}s)",
            status_code=504,
        )


class RemoteFetchHttpError(RemoteFetchError):
    """Remote server returned non-2xx status."""

    def __init__(self, url_host: str, http_status: int):
        super().__init__(
            f"Remote server {url_host} returned HTTP {http_status}",
            status_code=502,
        )
        self.http_status = http_status


class RemoteFetchTooLarge(RemoteFetchError):
    """Downloaded content exceeds size limit."""

    def __init__(self, size: int, max_size: int):
        super().__init__(
            f"Remote file too large: {size} bytes > {max_size} bytes limit",
            status_code=413,
        )


class RemoteFetchForbiddenHost(RemoteFetchError):
    """URL host is not in the allowlist."""

    def __init__(self, host: str):
        super().__init__(
            f"Host '{host}' is not in the allowed hosts list for remote fetch",
            status_code=403,
        )


class RemoteFetchRedirectError(RemoteFetchError):
    """Redirect chain violated policy (too many hops or forbidden host)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=403)


class RemoteFetchSSRFBlocked(RemoteFetchError):
    """URL targets a private/internal network address (SSRF protection)."""

    def __init__(self, host: str):
        super().__init__(
            f"Host '{host}' resolves to a private/internal address (SSRF blocked)",
            status_code=403,
        )


# ── Result ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FetchResult:
    """Successful fetch result."""

    content: bytes
    content_type: str | None
    content_length: int
    inferred_filename: str | None


# ── Helpers ───────────────────────────────────────────────────────────

def _mask_url(url: str) -> str:
    """Return scheme + host + path only — strip query params / fragments."""
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.hostname}{parsed.path}"
    except Exception:
        return "<unparseable-url>"


def _infer_filename(url: str, content_type: str | None) -> str | None:
    """Try to extract a filename from the URL path."""
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        if "/" in path:
            candidate = path.rsplit("/", 1)[-1]
            if "." in candidate and len(candidate) <= 255:
                return candidate
    except Exception:
        pass
    return None


def _parse_allowed_hosts(raw: str) -> frozenset[str]:
    """Parse comma-separated host list into a frozenset of lowercase hostnames."""
    if not raw or not raw.strip():
        return frozenset()
    return frozenset(
        h.strip().lower()
        for h in raw.split(",")
        if h.strip()
    )


# ── SSRF protection ─────────────────────────────────────────────────

# Hostnames that are always blocked regardless of allowlist
_SSRF_BLOCKED_HOSTNAMES: frozenset[str] = frozenset({
    "localhost",
    "metadata.google.internal",          # GCP metadata
    "metadata.google.internal.",
})


def _is_ssrf_dangerous(hostname: str) -> bool:
    """
    Check if a hostname is a known SSRF target.

    Blocks:
      - localhost, 127.x.x.x, ::1
      - link-local (169.254.x.x, fe80::)
      - AWS metadata (169.254.169.254)
      - GCP metadata (metadata.google.internal)
      - Private networks (10.x, 172.16-31.x, 192.168.x)
    """
    lower = hostname.lower()

    # Check blocked hostnames
    if lower in _SSRF_BLOCKED_HOSTNAMES:
        return True

    # Check IP addresses
    try:
        addr = ipaddress.ip_address(lower)
        if addr.is_loopback:        # 127.x, ::1
            return True
        if addr.is_private:         # 10.x, 172.16-31.x, 192.168.x
            return True
        if addr.is_link_local:      # 169.254.x.x, fe80::
            return True
        if addr.is_reserved:
            return True
    except ValueError:
        # Not a raw IP — that's fine, it's a hostname
        pass

    return False


# ── Host validation ──────────────────────────────────────────────────

def validate_fetch_url(
    url: str,
    *,
    allowed_hosts: frozenset[str] | None = None,
    enforce: bool = False,
    provider: str | None = None,
    check_ssrf: bool = True,
) -> str:
    """
    Validate a remote fetch URL against the allowlist and SSRF rules.

    Args:
        url: The URL to validate.
        allowed_hosts: Set of allowed hostnames (lowercase).
        enforce: If True, reject URLs whose host is not in the allowlist.
                 If False, only log a warning.
        provider: Optional provider name for context in error messages.
        check_ssrf: If True, block known SSRF-dangerous hosts/IPs.

    Returns:
        The validated URL (unchanged).

    Raises:
        RemoteFetchForbiddenHost: If enforce=True and host not in allowlist.
        RemoteFetchSSRFBlocked: If host is a known SSRF target.
        RemoteFetchError: If URL is malformed (no scheme/host).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        raise RemoteFetchError(
            "Malformed URL: unable to parse",
            status_code=400,
        )

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise RemoteFetchError(
            f"URL scheme '{scheme}' is not allowed (http/https only)",
            status_code=400,
        )

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise RemoteFetchError(
            "URL has no hostname",
            status_code=400,
        )

    # ── SSRF check (always enforced) ──────────────────────────────
    if check_ssrf and _is_ssrf_dangerous(hostname):
        logger.warning(
            "remote_fetch.ssrf_blocked host=%s provider=%s",
            hostname, provider,
        )
        raise RemoteFetchSSRFBlocked(hostname)

    # ── Allowlist check ───────────────────────────────────────────
    # Skip if no hosts configured or enforcement is off
    if not allowed_hosts or not enforce:
        if allowed_hosts and not enforce and hostname not in allowed_hosts:
            logger.warning(
                "remote_fetch.host_not_in_allowlist host=%s provider=%s "
                "enforce=false (audit_only)",
                hostname, provider,
            )
        return url

    # Enforce allowlist
    if hostname not in allowed_hosts:
        logger.warning(
            "remote_fetch.host_rejected host=%s provider=%s "
            "allowed_hosts_count=%d",
            hostname, provider, len(allowed_hosts),
        )
        raise RemoteFetchForbiddenHost(hostname)

    return url


# ── Fetcher ───────────────────────────────────────────────────────────

class RemoteFileFetcher:
    """
    Async fetcher for remote files (File-Service temporary URLs).

    Phase 1.1 hardening:
      - Validates URL host against allowlist before fetch.
      - Manual redirect following — re-validates host on EVERY hop.
      - SSRF protection — blocks private/internal IPs and metadata endpoints.
      - Configurable max redirect hops.

    Usage::

        fetcher = RemoteFileFetcher()
        result = await fetcher.fetch(url, provider="file-service")
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_size: int = DEFAULT_MAX_SIZE_BYTES,
        max_redirects: int | None = None,
    ):
        self._timeout = timeout
        self._max_size = max_size
        self._max_redirects = max_redirects  # None = load from settings

    @staticmethod
    def _get_allowed_hosts() -> frozenset[str]:
        """Load allowed hosts from settings (lazy, no import-time side effects)."""
        from app.core.config import settings

        return _parse_allowed_hosts(
            getattr(settings, "REMOTE_FETCH_ALLOWED_HOSTS", ""),
        )

    @staticmethod
    def _get_enforce_flag() -> bool:
        """Load enforcement flag from settings."""
        from app.core.config import settings

        return bool(getattr(settings, "REMOTE_FETCH_ENFORCE_ALLOWLIST", False))

    def _get_max_redirects(self) -> int:
        """Load max redirects from settings or use constructor override."""
        if self._max_redirects is not None:
            return self._max_redirects
        from app.core.config import settings

        return int(getattr(settings, "REMOTE_FETCH_MAX_REDIRECTS", DEFAULT_MAX_REDIRECTS))

    async def fetch(
        self,
        url: str,
        *,
        provider: str | None = None,
    ) -> FetchResult:
        """
        Download file from *url* with redirect-safe host validation.

        Phase 1.1: disables httpx auto-follow and manually follows
        redirects, validating each hop against the allowlist.

        Raises:
            RemoteFetchForbiddenHost — host not in allowlist (when enforced)
            RemoteFetchSSRFBlocked   — host targets private/internal network
            RemoteFetchRedirectError — redirect chain too long or to bad host
            RemoteFetchTimeout       — on connect / read timeout
            RemoteFetchHttpError     — on non-2xx, non-redirect response
            RemoteFetchTooLarge      — when body exceeds max_size (mid-stream)
            RemoteFetchError         — on any other transport error
        """
        allowed_hosts = self._get_allowed_hosts()
        enforce = self._get_enforce_flag()
        max_redirects = self._get_max_redirects()

        # ── Validate initial URL ──────────────────────────────────────
        validate_fetch_url(
            url,
            allowed_hosts=allowed_hosts,
            enforce=enforce,
            provider=provider,
        )

        safe_url = _mask_url(url)
        current_url = url
        redirect_count = 0

        logger.info("remote_fetch.start url=%s provider=%s", safe_url, provider)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=False,  # Phase 1.1: manual redirect handling
            ) as client:

                while True:
                    async with client.stream("GET", current_url) as response:
                        # ── Handle redirects manually ─────────────────
                        if response.status_code in (301, 302, 303, 307, 308):
                            redirect_count += 1
                            if redirect_count > max_redirects:
                                raise RemoteFetchRedirectError(
                                    f"Too many redirects ({redirect_count} > "
                                    f"{max_redirects})"
                                )

                            location = response.headers.get("location")
                            if not location:
                                raise RemoteFetchRedirectError(
                                    "Redirect response missing Location header"
                                )

                            # Validate redirect target host
                            try:
                                validate_fetch_url(
                                    location,
                                    allowed_hosts=allowed_hosts,
                                    enforce=enforce,
                                    provider=provider,
                                )
                            except (RemoteFetchForbiddenHost, RemoteFetchSSRFBlocked) as exc:
                                logger.warning(
                                    "remote_fetch.redirect_blocked "
                                    "from=%s to=%s hop=%d reason=%s",
                                    _mask_url(current_url),
                                    _mask_url(location),
                                    redirect_count,
                                    type(exc).__name__,
                                )
                                raise RemoteFetchRedirectError(
                                    f"Redirect to forbidden host at hop {redirect_count}: "
                                    f"{type(exc).__name__}"
                                )

                            logger.debug(
                                "remote_fetch.redirect hop=%d to=%s",
                                redirect_count, _mask_url(location),
                            )
                            current_url = location
                            continue  # Follow the redirect

                        # ── Non-redirect: validate status ─────────────
                        if response.status_code < 200 or response.status_code >= 300:
                            url_host = urlparse(current_url).hostname or safe_url
                            logger.warning(
                                "remote_fetch.http_error url=%s status=%d",
                                _mask_url(current_url), response.status_code,
                            )
                            raise RemoteFetchHttpError(url_host, response.status_code)

                        # ── Stream body with size guard ───────────────
                        chunks: list[bytes] = []
                        total = 0

                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > self._max_size:
                                raise RemoteFetchTooLarge(total, self._max_size)
                            chunks.append(chunk)

                        data = b"".join(chunks)
                        break  # Success — exit the redirect loop

        except (
            RemoteFetchHttpError,
            RemoteFetchTooLarge,
            RemoteFetchForbiddenHost,
            RemoteFetchRedirectError,
            RemoteFetchSSRFBlocked,
        ):
            raise

        except httpx.TimeoutException:
            url_host = urlparse(current_url).hostname or safe_url
            logger.warning("remote_fetch.timeout url=%s timeout=%s", safe_url, self._timeout)
            raise RemoteFetchTimeout(url_host, self._timeout)

        except httpx.HTTPError as exc:
            url_host = urlparse(current_url).hostname or safe_url
            logger.warning("remote_fetch.transport_error url=%s error=%s", safe_url, type(exc).__name__)
            raise RemoteFetchError(
                f"Failed to download file from {url_host}: {type(exc).__name__}",
                status_code=502,
            ) from exc

        if not data:
            url_host = urlparse(current_url).hostname or safe_url
            raise RemoteFetchError(
                f"Remote file from {url_host} is empty (0 bytes)",
                status_code=422,
            )

        content_type = response.headers.get("content-type")
        filename = _infer_filename(url, content_type)

        logger.info(
            "remote_fetch.done url=%s size=%d content_type=%s filename=%s redirects=%d",
            safe_url, len(data), content_type, filename, redirect_count,
        )

        return FetchResult(
            content=data,
            content_type=content_type,
            content_length=len(data),
            inferred_filename=filename,
        )
