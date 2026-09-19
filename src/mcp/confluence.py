# src/mcp/confluence.py
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup

from src.mcp.base import MCPSource, SourceDocument


class ConfluenceSource(MCPSource):
    def __init__(self, url: str, user: str, api_token: str, space_key: str):
        base = url.rstrip("/")
        self._base = f"{base}/wiki/rest/api"
        self._auth = HTTPBasicAuth(user, api_token)
        self._space_key = space_key

    def _get(self, path: str, params: dict = None) -> dict:
        response = requests.get(
            f"{self._base}{path}",
            params=params,
            auth=self._auth,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        keywords = [w for w in query.replace("?", "").replace(",", "").split() if len(w) > 2]
        keyword_clauses = " OR ".join(f'text ~ "{kw}"' for kw in keywords)
        cql = f'space = "{self._space_key}" AND ({keyword_clauses})'
        data = self._get("/content/search", params={"cql": cql, "limit": limit, "expand": "body.storage"})

        docs = []
        for r in data.get("results", []):
            page_id = r.get("id", "")
            title = r.get("title", "")
            html = r.get("body", {}).get("storage", {}).get("value", "")
            text = _html_to_text(html) if html else ""

            docs.append(SourceDocument(
                text=text,
                doc_id=page_id,
                metadata={
                    "source": "confluence",
                    "page_id": page_id,
                    "title": title,
                    "space_key": self._space_key,
                },
            ))

        return docs

    def fetch_page(self, page_id: str) -> SourceDocument:
        r = self._get(f"/content/{page_id}", params={"expand": "body.storage"})
        html = r.get("body", {}).get("storage", {}).get("value", "")
        return SourceDocument(
            text=_html_to_text(html),
            doc_id=page_id,
            metadata={
                "source": "confluence",
                "page_id": page_id,
                "title": r.get("title", ""),
                "space_key": self._space_key,
            },
        )


def _html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()


