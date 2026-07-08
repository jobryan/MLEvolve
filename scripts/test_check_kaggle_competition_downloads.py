#!/usr/bin/env python3
"""Tests for check_kaggle_competition_downloads.py."""

from __future__ import annotations

import importlib.util
import sys
import urllib.error
from email.message import Message
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_kaggle_competition_downloads.py"
spec = importlib.util.spec_from_file_location("check_kaggle_competition_downloads", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeResponse:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self._status = status
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value
        self.closed = False

    def getcode(self) -> int:
        return self._status

    def close(self) -> None:
        self.closed = True


def test_competition_ids_from_mixed_records() -> None:
    records = [
        {"task_id": "root-task"},
        {"task": {"id": "nested-task"}},
        {"parameters": {"task_id": "parameter-task"}},
        {"tags": {"task": "tag-task"}},
        {"containerOverrides": {"environment": [{"name": "ABLATION_TASK_ID", "value": "env-task"}]}},
        {"unrelated": True},
    ]
    assert module.competition_ids_from_records(records) == [
        "env-task",
        "nested-task",
        "parameter-task",
        "root-task",
        "tag-task",
    ]


def test_check_download_success() -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        assert timeout == 7
        assert request.headers["Authorization"].startswith("Basic ")
        assert request.headers["Range"] == "bytes=0-0"
        return FakeResponse(206, {"Content-Type": "application/zip", "Content-Range": "bytes 0-0/10"})

    result = module.check_download("spooky-author-identification", "user", "key", timeout=7, urlopen=fake_urlopen)
    assert result.ok is True
    assert result.status == 206
    assert result.content_range == "bytes 0-0/10"


def test_check_download_rules_not_accepted() -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            fp=FakeBody(b'{"message":"You must accept this competition rules"}'),
        )

    result = module.check_download("spooky-author-identification", "user", "key", timeout=7, urlopen=fake_urlopen)
    assert result.ok is False
    assert result.status == 403
    assert "accept" in result.body


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self, size: int = -1) -> bytes:
        return self._data if size < 0 else self._data[:size]

    def close(self) -> None:
        pass


if __name__ == "__main__":
    test_competition_ids_from_mixed_records()
    test_check_download_success()
    test_check_download_rules_not_accepted()
    print("check_kaggle_competition_downloads tests passed")
