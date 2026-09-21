from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import AsyncClient, AsyncClientOptions, Client, create_async_client, create_client
from supabase_auth.errors import AuthApiError
from supabase_auth.types import User

from core.config import get_settings
from utils.log import logger


def _get_supabase_credentials() -> tuple[str, str]:
    """Retrieve Supabase URL and publishable key from Settings."""
    settings = get_settings()
    url = settings.supabase_url
    key = settings.supabase_api_key
    if not url or not key:
        logger.error("Supabase URL or publishable key is missing in Settings.")
        raise ValueError("SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY must be set in environment variables.")
    return url, key


# Global singleton instances
_supabase_client: Client | None = None
_async_supabase_client: AsyncClient | None = None
_async_service_supabase_client: AsyncClient | None = None

# Attempt module-level sync client initialization on load if credentials exist
try:
    _url, _key = _get_supabase_credentials()
    _supabase_client = create_client(_url, _key)
    logger.info("Supabase sync client initialized")
except Exception as _e:
    logger.warning("Supabase client deferred initialization: %s", _e)

# Module-level client export for legacy/direct script usage
supabase: Client | None = _supabase_client


def get_supabase() -> Client:
    """FastAPI dependency to retrieve the synchronous Supabase client.
    
    Usage:
        @router.get("/items")
        def read_items(supabase: SupabaseDep):
            ...
    """
    global _supabase_client
    if _supabase_client is None:
        try:
            url, key = _get_supabase_credentials()
            _supabase_client = create_client(url, key)
            logger.info("Supabase sync client initialized")
        except Exception as e:
            logger.error("Failed to initialize Supabase client: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Supabase client not initialized: {e}",
            ) from e
    return _supabase_client


async def get_async_supabase() -> AsyncClient:
    """FastAPI dependency to retrieve the asynchronous Supabase client.
    
    Usage:
        @router.get("/items")
        async def read_items(supabase: AsyncSupabaseDep):
            ...
    """
    global _async_supabase_client
    if _async_supabase_client is None:
        try:
            url, key = _get_supabase_credentials()
            _async_supabase_client = await create_async_client(url, key)
            logger.info("Supabase async client initialized")
        except Exception as e:
            logger.error("Failed to initialize Supabase async client: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Supabase async client not initialized: {e}",
            ) from e
    return _async_supabase_client


# FastAPI Annotated dependency aliases per best practices
SupabaseDep = Annotated[Client, Depends(get_supabase)]
SupabaseClient = SupabaseDep

AsyncSupabaseDep = Annotated[AsyncClient, Depends(get_async_supabase)]
AsyncSupabaseClient = AsyncSupabaseDep

# Bearer token security for authentication dependency
_bearer_security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_security)],
    client: SupabaseDep,
) -> User:
    """FastAPI dependency to authenticate requests using a Supabase JWT bearer token.
    
    Usage:
        @router.get("/profile")
        def get_profile(user: CurrentUserDep):
            return {"user_id": user.id, "email": user.email}
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        user_response = client.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_response.user
    except AuthApiError as e:
        logger.warning("Supabase auth error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired authentication token: {e.message}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except Exception as e:
        logger.error("Unexpected error validating token: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


CurrentUserDep = Annotated[User, Depends(get_current_user)]
CurrentUser = CurrentUserDep


def _get_secret_key() -> str:
    """Backend-only key. Prefer sb_secret_ over the legacy JWT service_role key."""
    key = get_settings().supabase_admin_key
    if not key:
        logger.error("Supabase secret key is missing in Settings.")
        raise ValueError("SUPABASE_SECRET_KEY must be set in environment variables.")
    return key


async def get_async_service_supabase() -> AsyncClient:
    """Trusted backend client. Use after JWT auth; always filter writes by owner."""
    global _async_service_supabase_client
    if _async_service_supabase_client is None:
        try:
            url, _publishable_key = _get_supabase_credentials()
            secret_key = _get_secret_key()
            _async_service_supabase_client = await create_async_client(
                url,
                secret_key,
                options=AsyncClientOptions(
                    persist_session=False,
                    auto_refresh_token=False,
                ),
            )
            logger.info("Supabase async secret-key client initialized")
        except Exception as e:
            logger.error("Failed to initialize Supabase secret-key client: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Supabase service client not initialized: {e}",
            ) from e
    return _async_service_supabase_client


async def get_async_user_supabase(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_security)],
) -> AsyncIterator[AsyncClient]:
    """Per-request client that carries the caller's JWT so RLS applies."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        url, key = _get_supabase_credentials()
        client = await create_async_client(
            url,
            key,
            options=AsyncClientOptions(
                persist_session=False,
                auto_refresh_token=False,
                headers={"Authorization": f"Bearer {credentials.credentials}", "apikey": key},
            ),
        )
    except Exception as e:
        logger.error("Failed to initialize user-scoped Supabase client: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Supabase user client not initialized: {e}",
        ) from e
    yield client


AsyncServiceSupabaseDep = Annotated[AsyncClient, Depends(get_async_service_supabase)]
AsyncUserSupabaseDep = Annotated[AsyncClient, Depends(get_async_user_supabase)]
