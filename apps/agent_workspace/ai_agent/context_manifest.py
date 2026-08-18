import errno
import hashlib
import json
import os
import posixpath
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from apps.agent_workspace.exceptions import InvalidContextManifest

END_USER_CONTEXT_ROOT = Path(__file__).resolve().parents[1] / "end_user_context"
DEFAULT_CONTEXT_MANIFEST_PATH = END_USER_CONTEXT_ROOT / "context_manifest.json"


@dataclass(frozen=True)
class ContextFileMapping:
    """
    Represent one validated source file and runtime destination mapping.
    """

    source: str
    destination: str
    source_path: Path
    content: bytes


@dataclass(frozen=True)
class ContextManifest:
    """
    Represent a validated end-user context manifest and its fingerprint.
    """

    context_version: str
    files: tuple[ContextFileMapping, ...]
    context_hash: str


def load_context_manifest(
    manifest_path: Path | None = None,
    context_root: Path | None = None,
) -> ContextManifest:
    """
    Load, validate, and fingerprint the end-user context manifest.

    Source files are validated against the end-user context root. Runtime
    destinations are normalized as relative POSIX paths and checked for
    duplicates after normalization.

    Raises:
        InvalidContextManifest: If the manifest is malformed or unsafe.
    """

    root_path = Path(context_root or END_USER_CONTEXT_ROOT)
    manifest_file_path = Path(manifest_path or DEFAULT_CONTEXT_MANIFEST_PATH)
    raw_manifest = _load_json_manifest(manifest_file_path)
    context_version = _validate_context_version(raw_manifest)
    files = _validate_manifest_files(raw_manifest, root_path)
    context_hash = _compute_context_hash(context_version, files)

    return ContextManifest(
        context_version=context_version,
        files=files,
        context_hash=context_hash,
    )


def _load_json_manifest(manifest_path: Path) -> dict[str, Any]:
    """
    Read and parse a JSON context manifest file.
    """

    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise InvalidContextManifest("Manifest file cannot be read") from exc

    try:
        manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidContextManifest("Manifest JSON is malformed") from exc

    if not isinstance(manifest_data, dict):
        raise InvalidContextManifest("Manifest root must be an object")

    return manifest_data


def _validate_context_version(manifest_data: dict[str, Any]) -> str:
    """
    Validate and return the explicit context version.
    """

    context_version = manifest_data.get("context_version")

    if not isinstance(context_version, str) or context_version.strip() == "":
        raise InvalidContextManifest("Context version must be a non-empty string")

    return context_version


def _validate_manifest_files(
    manifest_data: dict[str, Any],
    context_root: Path,
) -> tuple[ContextFileMapping, ...]:
    """
    Validate manifest file entries and reject duplicate destinations.
    """

    files_data = manifest_data.get("files")

    if not isinstance(files_data, list) or len(files_data) == 0:
        raise InvalidContextManifest("Files must be a non-empty list")

    root_path = _resolve_context_root(context_root)
    mappings: list[ContextFileMapping] = []
    destinations: set[str] = set()

    for index, file_entry in enumerate(files_data):
        if not isinstance(file_entry, dict):
            raise InvalidContextManifest(f"Files[{index}] must be an object")

        source = file_entry.get("source")
        destination = file_entry.get("target")

        if not isinstance(source, str) or source.strip() == "":
            raise InvalidContextManifest(f"Files[{index}].source must be a non-empty string")

        if not isinstance(destination, str) or destination.strip() == "":
            raise InvalidContextManifest(f"Files[{index}].target must be a non-empty string")

        normalized_source = _normalize_relative_path(source, f"Files[{index}].source")
        normalized_destination = _normalize_relative_path(destination, f"Files[{index}].target")

        if normalized_destination in destinations:
            raise InvalidContextManifest(f"Duplicate destination path: {normalized_destination}")

        source_path, content = _validate_source_file(root_path, normalized_source)
        destinations.add(normalized_destination)
        mappings.append(
            ContextFileMapping(
                source=normalized_source,
                destination=normalized_destination,
                source_path=source_path,
                content=content,
            )
        )

    return tuple(mappings)


def _resolve_context_root(context_root: Path) -> Path:
    """
    Resolve the configured end-user context root.
    """

    try:
        return context_root.resolve(strict=True)
    except OSError as exc:
        raise InvalidContextManifest("Context root cannot be resolved") from exc


def _normalize_relative_path(path_value: str, field_name: str) -> str:
    """
    Normalize a manifest path and ensure it remains relative.
    """

    if "\x00" in path_value:
        raise InvalidContextManifest(f"{field_name} contains a null byte")

    if "\\" in path_value:
        raise InvalidContextManifest(f"{field_name} must use POSIX path separators")

    posix_path = PurePosixPath(path_value)
    windows_path = PureWindowsPath(path_value)

    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise InvalidContextManifest(f"{field_name} must be relative")

    if ".." in posix_path.parts:
        raise InvalidContextManifest(f"{field_name} must not traverse outside context")

    normalized_path = posixpath.normpath(path_value)

    if normalized_path in {"", "."}:
        raise InvalidContextManifest(f"{field_name} must identify a file path")

    normalized_posix_path = PurePosixPath(normalized_path)

    if normalized_posix_path.is_absolute():
        raise InvalidContextManifest(f"{field_name} must not traverse outside context")

    return normalized_path


def _validate_source_file(context_root: Path, normalized_source: str) -> tuple[Path, bytes]:
    """
    Validate that a source path is a regular file and snapshot its bytes.
    """

    source_path = context_root.joinpath(*PurePosixPath(normalized_source).parts)

    if source_path.is_symlink():
        raise InvalidContextManifest(f"Source must not be a symlink: {normalized_source}")

    if not source_path.exists():
        raise InvalidContextManifest(f"Source file does not exist: {normalized_source}")

    if not source_path.is_file():
        raise InvalidContextManifest(f"Source must be a regular file: {normalized_source}")

    try:
        resolved_source_path = source_path.resolve(strict=True)
        resolved_source_path.relative_to(context_root)
    except (OSError, ValueError) as exc:
        raise InvalidContextManifest(f"Source escapes context root: {normalized_source}") from exc

    content = _read_source_file_snapshot(context_root, normalized_source)

    return resolved_source_path, content


def _read_source_file_snapshot(context_root: Path, normalized_source: str) -> bytes:
    """
    Read source bytes from a regular file opened under the context root.
    """

    source_fd: int | None = None

    try:
        source_fd = _open_context_source_file(context_root, normalized_source)
        source_stat = os.fstat(source_fd)

        if not stat.S_ISREG(source_stat.st_mode):
            raise InvalidContextManifest(f"Source must be a regular file: {normalized_source}")

        with os.fdopen(source_fd, "rb") as source_file:
            source_fd = None

            return source_file.read()
    except OSError as exc:
        raise _source_file_open_error(normalized_source, exc) from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)


def _open_context_source_file(context_root: Path, normalized_source: str) -> int:
    """
    Open a context source through directory file descriptors without symlinks.
    """

    parts = PurePosixPath(normalized_source).parts
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow_flag
    file_flags = os.O_RDONLY | nofollow_flag
    root_fd = os.open(context_root, directory_flags)
    current_directory_fd = root_fd

    try:
        for directory_part in parts[:-1]:
            next_directory_fd = os.open(
                directory_part,
                directory_flags,
                dir_fd=current_directory_fd,
            )

            if current_directory_fd != root_fd:
                os.close(current_directory_fd)

            current_directory_fd = next_directory_fd

        return os.open(parts[-1], file_flags, dir_fd=current_directory_fd)
    finally:
        os.close(current_directory_fd)

        if current_directory_fd != root_fd:
            os.close(root_fd)


def _source_file_open_error(normalized_source: str, exc: OSError) -> InvalidContextManifest:
    """
    Convert source snapshot open failures into deterministic manifest errors.
    """

    if isinstance(exc, FileNotFoundError):
        return InvalidContextManifest(f"Source file does not exist: {normalized_source}")

    if isinstance(exc, NotADirectoryError):
        return InvalidContextManifest(f"Source must be a regular file: {normalized_source}")

    if exc.errno == errno.ELOOP:
        return InvalidContextManifest(f"Source must not be a symlink: {normalized_source}")

    return InvalidContextManifest(f"Source file cannot be read: {normalized_source}")


def _compute_context_hash(
    context_version: str,
    files: tuple[ContextFileMapping, ...],
) -> str:
    """
    Compute a deterministic SHA-256 fingerprint for context contents.
    """

    digest = hashlib.sha256()
    _update_length_prefixed(digest, b"agent-context-manifest-v1")
    _update_length_prefixed(digest, context_version.encode("utf-8"))

    for file_mapping in sorted(files, key=lambda mapping: mapping.destination):
        _update_length_prefixed(digest, file_mapping.destination.encode("utf-8"))
        _update_length_prefixed(digest, file_mapping.content)

    return digest.hexdigest()


def _update_length_prefixed(digest, value: bytes) -> None:
    """
    Add unambiguous length-prefixed bytes to a hash digest.
    """

    digest.update(str(len(value)).encode("ascii"))
    digest.update(b":")
    digest.update(value)
    digest.update(b"\n")
