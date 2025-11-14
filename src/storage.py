from __future__ import annotations

from typing import Any, Dict, List, Optional
import json


class QdrantStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "announcements",
        vector_size: int = 768,
        distance: str = "Cosine",
        api_key: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.collection = collection
        self.vector_size = vector_size
        self.distance = distance
        self.api_key = api_key
        self._client = None

    def _client_or_connect(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient  # type: ignore
                from qdrant_client.http import models as rest  # noqa: F401
            except Exception as e:
                raise RuntimeError("qdrant-client is required for Qdrant storage.") from e
            self._client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
        return self._client

    def ensure_collection(self) -> None:
        from qdrant_client.http import models as rest  # type: ignore
        client = self._client_or_connect()
        collections = client.get_collections().collections
        names = {c.name for c in collections}
        if self.collection not in names:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(size=self.vector_size, distance=self.distance),
            )

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        from qdrant_client.http import models as rest  # type: ignore
        client = self._client_or_connect()
        self.ensure_collection()
        client.upsert(
            collection_name=self.collection,
            points=[rest.PointStruct(id=pid, vector=vec, payload=pl) for pid, vec, pl in zip(ids, vectors, payloads)],
        )


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def upsert_record(self, record: Dict[str, Any]) -> None:
        try:
            import psycopg2  # type: ignore
        except Exception as e:
            raise RuntimeError("psycopg2-binary is required for Postgres storage.") from e
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS announcements (
                        ref TEXT,
                        date TEXT,
                        language TEXT,
                        announcement TEXT,
                        source_pdf TEXT,
                        metadata JSONB
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO announcements (ref, date, language, announcement, source_pdf, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                    """,
                    (
                        record.get("ref"),
                        record.get("date"),
                        record.get("language"),
                        record.get("announcement"),
                        record.get("source_pdf"),
                        json.dumps(record.get("metadata") or {}),
                    ),
                )


__all__ = ["QdrantStore", "PostgresStore"]
