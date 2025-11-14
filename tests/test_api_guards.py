import io
import time

import pytest
from starlette.datastructures import UploadFile
from fastapi import HTTPException

from src.server import (
    _check_rate_limit,
    _request_log,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    _validate_upload,
    MAX_FILE_SIZE_MB,
)


def make_upload(filename: str, content: bytes, content_type: str = "application/pdf"):
    return UploadFile(filename=filename, file=io.BytesIO(content), content_type=content_type)


def test_validate_upload_rejects_non_pdf():
    upload = make_upload("note.txt", b"hello", content_type="text/plain")
    with pytest.raises(HTTPException) as exc:
        _validate_upload(upload, b"hello")
    assert exc.value.status_code == 400
    assert "PDF" in exc.value.detail


def test_validate_upload_rejects_large_file():
    big_size = (MAX_FILE_SIZE_MB * 1024 * 1024) + 1
    content = b"0" * big_size
    upload = make_upload("huge.pdf", content)
    with pytest.raises(HTTPException) as exc:
        _validate_upload(upload, content)
    assert exc.value.status_code == 400
    assert "too large" in exc.value.detail.lower()


def test_rate_limit_blocks_after_threshold(monkeypatch):
    _request_log.clear()
    ip = "1.2.3.4"

    # Hit the limit
    for _ in range(RATE_LIMIT_REQUESTS):
        _check_rate_limit(ip)

    with pytest.raises(HTTPException) as exc:
        _check_rate_limit(ip)
    assert exc.value.status_code == 429

    # Move time forward to release window
    future = time.time() + RATE_LIMIT_WINDOW_SECONDS + 1

    def fake_time():
        return future

    monkeypatch.setattr("src.server.time", fake_time)
    _check_rate_limit(ip)
*** End of File