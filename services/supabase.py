import os
from typing import Annotated
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import AsyncClient, Client, create_async_client, create_client
from supabase_auth.errors import AuthApiError
from supabase_auth.types import User

from utils.log import logger

load_dotenv()


def _get_supabase_credentials() -> tuple[str, str]:
    """Retrieve Supabase URL and API Key from environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE__PUBLISHABLE_KEY")
    )
    if not url or not key:
        logger.error("Supabase URL or Publishable/Anon Key is missing in environment variables.")
        raise ValueError("SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY must be set in environment variables.")
    return url, key


# Global singleton instances
_supabase_client: Client | None = None
_async_supabase_client: AsyncClient | None = None

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
