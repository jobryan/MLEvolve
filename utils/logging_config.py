import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class VerboseFilter(logging.Filter):
    """Filter out records marked with the verbose attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (hasattr(record, "verbose") and record.verbose)


class S3LogBackupManager:
    """Archives active MLEvolve log files to S3 and truncates local copies."""

    def __init__(
        self,
        *,
        log_dir: Path,
        destination_uri: str,
        threshold_bytes: int,
        check_interval_seconds: float,
        handlers: list[logging.FileHandler],
    ) -> None:
        self.log_dir = log_dir
        self.destination_uri = destination_uri.rstrip("/")
        self.threshold_bytes = threshold_bytes
        self.check_interval_seconds = check_interval_seconds
        self.handlers = handlers
        self._lock = threading.RLock()
        self._next_check_at = 0.0
        self._last_error_at = 0.0

        without_scheme = self.destination_uri.removeprefix("s3://")
        self.bucket, _, raw_prefix = without_scheme.partition("/")
        self.prefix = raw_prefix.strip("/")
        if not self.bucket:
            raise ValueError(f"invalid S3 log backup URI: {destination_uri!r}")

    def maybe_archive(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._next_check_at:
            return

        with self._lock:
            now = time.monotonic()
            if not force and now < self._next_check_at:
                return
            self._next_check_at = now + self.check_interval_seconds

            log_paths = self._log_paths()
            total_size = sum(path.stat().st_size for path in log_paths if path.exists())
            if total_size < self.threshold_bytes:
                return

            acquired: list[logging.FileHandler] = []
            try:
                for handler in self.handlers:
                    handler.acquire()
                    acquired.append(handler)

                for handler in self.handlers:
                    if handler.stream:
                        handler.flush()

                self._upload_and_truncate(log_paths)
            except Exception as exc:  # pragma: no cover - depends on AWS/network.
                self._report_error(exc)
            finally:
                for handler in reversed(acquired):
                    handler.release()

    def _log_paths(self) -> list[Path]:
        paths: list[Path] = []
        for handler in self.handlers:
            filename = getattr(handler, "baseFilename", None)
            if filename:
                paths.append(Path(filename))
        return sorted(set(paths))

    def _upload_and_truncate(self, log_paths: list[Path]) -> None:
        archive_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for path in log_paths:
            if not path.exists() or path.stat().st_size == 0:
                continue
            key = self._object_key(path, archive_id)
            self._upload_file(path, key)

        for handler in self.handlers:
            if not handler.stream:
                continue
            handler.flush()
            handler.stream.truncate(0)
            handler.stream.seek(0)

    def _upload_file(self, path: Path, key: str) -> None:
        try:
            import boto3  # type: ignore[import-not-found]

            boto3.client("s3").upload_file(str(path), self.bucket, key)
            return
        except ModuleNotFoundError:
            pass

        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(path),
                f"s3://{self.bucket}/{key}",
                "--only-show-errors",
            ],
            check=True,
        )

    def _object_key(self, path: Path, archive_id: str) -> str:
        try:
            relative_path = path.relative_to(self.log_dir)
        except ValueError:
            relative_path = Path(path.name)
        key_parts = [
            part
            for part in (
                self.prefix,
                "log-archives",
                self._log_dir_label(),
                archive_id,
                str(relative_path),
            )
            if part
        ]
        return "/".join(key_parts)

    def _log_dir_label(self) -> str:
        run_name = self.log_dir.parent.name if self.log_dir.name == "logs" else self.log_dir.name
        digest = hashlib.sha1(str(self.log_dir.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"{run_name}-{digest}"

    def _report_error(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_at < 300:
            return
        self._last_error_at = now
        print(f"MLEvolve log backup failed; local logs were kept: {exc}", file=sys.stderr)


class ManagedFileHandler(logging.FileHandler):
    def __init__(self, filename: Path, backup_manager: S3LogBackupManager | None = None) -> None:
        super().__init__(filename)
        self.backup_manager = backup_manager

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if self.backup_manager is not None:
            self.backup_manager.maybe_archive()


def _get_cfg_value(cfg: Any, dotted_path: str, default: Any = None) -> Any:
    current = cfg
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def _log_backup_uri(cfg: Any) -> str:
    env_uri = os.environ.get("MLEVOLVE_LOG_BACKUP_URI") or os.environ.get("ABLATION_ARTIFACT_ROOT")
    if env_uri:
        return env_uri
    configured_uri = _get_cfg_value(cfg, "log_backup.uri", "")
    if configured_uri:
        return str(configured_uri)
    return ""


def _build_log_backup_manager(
    cfg: Any,
    handlers: list[logging.FileHandler],
) -> S3LogBackupManager | None:
    enabled = bool(_get_cfg_value(cfg, "log_backup.enabled", True))
    if not enabled:
        return None

    destination_uri = _log_backup_uri(cfg).strip()
    if not destination_uri:
        return None
    if not destination_uri.startswith("s3://"):
        print(
            f"MLEvolve log backup only supports s3:// URIs; got {destination_uri!r}",
            file=sys.stderr,
        )
        return None

    threshold_bytes = int(_get_cfg_value(cfg, "log_backup.threshold_bytes", 1024**3))
    check_interval_seconds = float(
        _get_cfg_value(cfg, "log_backup.check_interval_seconds", 30)
    )
    return S3LogBackupManager(
        log_dir=Path(cfg.log_dir),
        destination_uri=destination_uri,
        threshold_bytes=threshold_bytes,
        check_interval_seconds=check_interval_seconds,
        handlers=handlers,
    )


def _reset_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def setup_logging(cfg: Any) -> logging.Logger:
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format=log_format,
        handlers=[],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger("MLEvolve")
    logger.setLevel(getattr(logging, cfg.log_level.upper()))
    logger.propagate = False
    _reset_logger_handlers(logger)

    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = ManagedFileHandler(cfg.log_dir / "MLEvolve.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = ManagedFileHandler(cfg.log_dir / "MLEvolve.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    file_handlers: list[logging.FileHandler] = [file_handler, verbose_file_handler]
    backup_manager = _build_log_backup_manager(cfg, file_handlers)
    file_handler.backup_manager = backup_manager
    verbose_file_handler.backup_manager = backup_manager

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)
    return logger
