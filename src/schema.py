from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import json


try:
    from pydantic import BaseModel, Field

    class AnnouncementRecord(BaseModel):
        ref: Optional[str] = Field(default=None, description="Reference number identifier, e.g., '32/2022' or '№ 32/2022'")
        date: Optional[str] = Field(default=None, description="ISO or detected date string, e.g., '2022-06-12' or '12.06.2022'")
        language: Optional[str] = Field(default=None, description="BCP47-ish tag like 'uz-latin', 'uz-cyrillic', 'ru', 'en'")
        announcement: Optional[str] = Field(default=None, description="Clean, corrected announcement text body")
        source_pdf: Optional[str] = Field(default=None, description="Original PDF path")
        embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for semantic search")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata incl. processed_at, page mapping")
        schema_version: str = Field(default="v1")

        def to_json(self) -> str:
            return self.model_dump_json(ensure_ascii=False, indent=2)

except Exception:

    @dataclass
    class AnnouncementRecord:
        ref: Optional[str] = None
        date: Optional[str] = None
        language: Optional[str] = None
        announcement: Optional[str] = None
        source_pdf: Optional[str] = None
        embedding: Optional[List[float]] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
        schema_version: str = "v1"

        def model_dump(self) -> Dict[str, Any]:
            return asdict(self)

        def to_json(self) -> str:
            return json.dumps(self.model_dump(), ensure_ascii=False, indent=2)


__all__ = ["AnnouncementRecord"]
