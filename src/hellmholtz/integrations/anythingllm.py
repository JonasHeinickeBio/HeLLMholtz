"""AnythingLLM REST API client.

AnythingLLM exposes a JSON REST API at ``/api/v1/…``.  This module
provides a thin, typed wrapper that HeLLMholtz can use to:

* list workspaces
* upload documents to a workspace
* chat with a workspace (document-aware, RAG-backed)

Authentication is done via a Bearer API key that is read from the
environment variable ``ANYTHINGLLM_API_KEY``.  The server URL is read
from ``ANYTHINGLLM_BASE_URL`` (defaults to ``http://localhost:3001``).

Secrets are **never** logged, included in ``__repr__``, or passed as
positional arguments.  Always supply ``api_key`` as a keyword argument
(or rely on the env var) to avoid leaking it via shell history.

Typical usage::

    from hellmholtz.integrations.anythingllm import AnythingLLMClient

    client = AnythingLLMClient()          # reads env vars automatically
    workspaces = client.list_workspaces()
    client.upload_document("my-workspace", Path("report.pdf"))
    reply = client.chat("my-workspace", "What are the key findings?")
    print(reply)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:3001"
_DEFAULT_TIMEOUT = 60  # seconds


class AnythingLLMError(Exception):
    """Raised when the AnythingLLM API returns an error."""


class AnythingLLMClient:
    """Thin, typed client for the AnythingLLM REST API.

    Parameters
    ----------
    base_url:
        Base URL of the AnythingLLM instance.  Defaults to the value of
        ``ANYTHINGLLM_BASE_URL`` env var, or ``http://localhost:3001``.
    api_key:
        Bearer API key.  Defaults to the value of ``ANYTHINGLLM_API_KEY``
        env var.  Required for all calls except ``/api/ping``.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = (
            (base_url or os.environ.get("ANYTHINGLLM_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        )
        _key = api_key or os.environ.get("ANYTHINGLLM_API_KEY") or ""
        if not _key:
            logger.warning(
                "ANYTHINGLLM_API_KEY is not set; API calls will fail authentication."
            )
        # Store key in a private attribute; never include in __repr__ or logs.
        self.__api_key = _key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get(self, path: str) -> Any:
        url = f"{self._base_url}{path}"
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
        return self._handle_response(resp)

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self._base_url}{path}"
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        return self._handle_response(resp)

    @staticmethod
    def _handle_response(resp: requests.Response) -> Any:
        if not resp.ok:
            # Never log the response body wholesale – it may contain secrets.
            raise AnythingLLMError(
                f"AnythingLLM API error {resp.status_code}: {resp.reason} "
                f"(URL: {resp.url})"
            )
        if not resp.content:
            return {}
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return *True* if the AnythingLLM instance is reachable.

        This endpoint does not require authentication.
        """
        try:
            url = f"{self._base_url}/api/ping"
            resp = requests.get(url, timeout=self.timeout)
            return resp.ok
        except requests.RequestException:
            return False

    def list_workspaces(self) -> list[dict[str, Any]]:
        """Return all workspaces visible to the API key.

        Returns:
            List of workspace dicts with at least ``slug`` and ``name`` keys.
        """
        data = self._get("/api/v1/workspaces")
        return list(data.get("workspaces", []))

    def create_workspace(self, name: str) -> dict[str, Any]:
        """Create a new workspace and return its metadata.

        Args:
            name: Human-readable workspace name.

        Returns:
            The created workspace object (includes ``slug``).
        """
        data = self._post("/api/v1/workspace/new", {"name": name})
        return dict(data.get("workspace", data))

    def upload_document(self, workspace_slug: str, file_path: Path) -> dict[str, Any]:
        """Upload a local file to a workspace.

        Args:
            workspace_slug: The slug identifier of the target workspace.
            file_path:      Path to the local file (PDF, DOCX, TXT, MD …).

        Returns:
            API response dict (includes document metadata).

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            AnythingLLMError:  On API errors.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        url = f"{self._base_url}/api/v1/document/upload"
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Accept": "application/json",
            # Do NOT set Content-Type here; requests sets it for multipart/form-data.
        }

        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh)}
            resp = requests.post(url, headers=headers, files=files, timeout=self.timeout)

        data: dict[str, Any] = self._handle_response(resp)

        # After upload, embed the document in the workspace so it is retrievable.
        if "document" in data:
            doc_location = data["document"].get("location", "")
            if doc_location:
                self._embed_document(workspace_slug, doc_location)

        return data

    def _embed_document(self, workspace_slug: str, doc_location: str) -> None:
        """Embed an already-uploaded document into a workspace's vector store."""
        self._post(
            f"/api/v1/workspace/{workspace_slug}/update-embeddings",
            {"adds": [doc_location], "deletes": []},
        )

    def chat(
        self,
        workspace_slug: str,
        message: str,
        *,
        mode: str = "chat",
        session_id: str | None = None,
    ) -> str:
        """Send a message to a workspace and return the text reply.

        Args:
            workspace_slug: Target workspace slug.
            message:        User's message / question.
            mode:           ``"chat"`` (conversation history kept) or
                            ``"query"`` (one-shot RAG).
            session_id:     Optional conversation thread identifier.

        Returns:
            The LLM's plain-text reply string.
        """
        payload: dict[str, Any] = {"message": message, "mode": mode}
        if session_id:
            payload["sessionId"] = session_id

        data = self._post(f"/api/v1/workspace/{workspace_slug}/chat", payload)
        reply: str = data.get("textResponse", "")
        return reply

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"AnythingLLMClient(base_url={self._base_url!r})"
