import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from django.conf import settings

from apps.agent_workspace.ai_agent.context_manifest import load_context_manifest
from apps.agent_workspace.exceptions import (
    AgentWorkspaceError,
    WorkspaceCleanupError,
    WorkspaceIdentityError,
)

WORKSPACE_DIRECTORIES = ("inputs", "outputs", "runtime", "logs")
OPEN_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
PRIVATE_DIRECTORY_MODE = 0o700
UNTRUSTED_WRITE_BITS = stat.S_IWGRP | stat.S_IWOTH


@dataclass(frozen=True)
class FilesystemIdentity:
    """
    Represent a filesystem object identity.
    """

    device: int
    inode: int


@dataclass(frozen=True)
class PreparedWorkspace:
    """
    Represent a prepared AgentRun workspace and context audit values.
    """

    workspace_path: Path
    context_version: str
    context_hash: str


@dataclass(frozen=True)
class TrustedWorkspaceRoot:
    """
    Represent a pinned trusted workspace root directory.
    """

    path: Path
    fd: int

    def close(self) -> None:
        """
        Close the pinned workspace-root file descriptor.
        """

        os.close(self.fd)


class WorkspaceManager:
    """
    Manage per-AgentRun workspace path derivation, bootstrap, and cleanup.
    """

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        manifest_path: Path | str | None = None,
        context_root: Path | str | None = None,
    ):
        """
        Store filesystem roots used by workspace lifecycle operations.
        """

        configured_workspace_root = workspace_root or settings.AGENT_WORKSPACE_ROOT

        self.workspace_root = Path(configured_workspace_root).absolute()
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.context_root = Path(context_root) if context_root is not None else None
        self._unsafe_cleanup_workspace_names: set[str] = set()

    def get_workspace_path(self, run_id: UUID | str) -> Path:
        """
        Return the canonical per-run workspace path for an AgentRun identifier.
        """

        run_uuid = _normalize_run_uuid(run_id)

        return self.workspace_root / str(run_uuid)

    def prepare_workspace(self, run_id: UUID | str) -> PreparedWorkspace:
        """
        Create and bootstrap a fresh per-run workspace from the context manifest.

        Raises:
            AgentWorkspaceError: If the run identifier or target workspace is invalid.
            InvalidContextManifest: If the context manifest is invalid or unsafe.
        """

        run_uuid = _normalize_run_uuid(run_id)
        workspace_name = str(run_uuid)
        trusted_workspace_root = self._acquire_workspace_root(create=True)

        if trusted_workspace_root is None:
            raise AgentWorkspaceError("Workspace root cannot be created")

        try:
            workspace_path = trusted_workspace_root.path / workspace_name
            try:
                created_workspace_identity = _create_workspace_directory(
                    trusted_workspace_root.fd,
                    workspace_name,
                )

                with _open_workspace_directory(
                    trusted_workspace_root.fd,
                    workspace_name,
                    expected_identity=created_workspace_identity,
                ) as workspace_fd:
                    _assert_workspace_identity(
                        trusted_workspace_root.fd,
                        workspace_fd,
                        workspace_name,
                        expected_identity=created_workspace_identity,
                    )

                    try:
                        for directory_name in WORKSPACE_DIRECTORIES:
                            _create_workspace_subdirectory(workspace_fd, directory_name)

                        context_manifest = load_context_manifest(
                            manifest_path=self.manifest_path,
                            context_root=self.context_root,
                        )

                        for file_mapping in context_manifest.files:
                            _write_context_file_at(
                                workspace_fd,
                                file_mapping.destination,
                                file_mapping.content,
                            )

                        _assert_workspace_identity(
                            trusted_workspace_root.fd,
                            workspace_fd,
                            workspace_name,
                            expected_identity=created_workspace_identity,
                        )
                        _assert_workspace_root_identity(
                            trusted_workspace_root.path,
                            trusted_workspace_root.fd,
                        )
                    except WorkspaceIdentityError:
                        raise
                    except Exception as exc:
                        _assert_bootstrap_failure_preserved_identity(
                            trusted_workspace_root,
                            workspace_fd,
                            workspace_name,
                            expected_identity=created_workspace_identity,
                            original_exception=exc,
                        )

                        raise
            except WorkspaceIdentityError:
                self._unsafe_cleanup_workspace_names.add(workspace_name)
                raise

            return PreparedWorkspace(
                workspace_path=workspace_path,
                context_version=context_manifest.context_version,
                context_hash=context_manifest.context_hash,
            )
        finally:
            trusted_workspace_root.close()

    def cleanup_workspace(self, run_id: UUID | str) -> None:
        """
        Remove the per-run workspace without following a workspace symlink.

        Missing workspaces are treated as successful cleanup so callers may
        retry cleanup idempotently.

        Raises:
            AgentWorkspaceError: If the run identifier or workspace path is invalid.
        """

        run_uuid = _normalize_run_uuid(run_id)
        workspace_name = str(run_uuid)

        if workspace_name in self._unsafe_cleanup_workspace_names:
            return

        trusted_workspace_root = self._acquire_workspace_root(create=False)

        if trusted_workspace_root is None:
            return

        try:
            _remove_workspace_directory(trusted_workspace_root.fd, workspace_name)
        finally:
            trusted_workspace_root.close()

    def _acquire_workspace_root(self, *, create: bool) -> TrustedWorkspaceRoot | None:
        """
        Acquire the configured workspace root as a pinned directory descriptor.
        """

        return _acquire_workspace_root(self.workspace_root, create=create)


def cleanup_agent_run_workspace(run_id: UUID | str) -> None:
    """
    Attempt cleanup for a terminal AgentRun workspace.
    """

    WorkspaceManager().cleanup_workspace(run_id)


def _normalize_run_uuid(run_id: UUID | str) -> UUID:
    """
    Validate and normalize an AgentRun identifier into a canonical UUID.
    """

    try:
        return run_id if isinstance(run_id, UUID) else UUID(str(run_id))
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("AgentRun identifier must be a valid UUID") from exc


class _OpenDirectory:
    """
    Context manager for directory file descriptors.
    """

    def __init__(self, directory_fd: int):
        """
        Store a directory descriptor to close when leaving the context.
        """

        self.directory_fd = directory_fd

    def __enter__(self) -> int:
        """
        Return the managed directory descriptor.
        """

        return self.directory_fd

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Close the managed directory descriptor.
        """

        os.close(self.directory_fd)


def _acquire_workspace_root(path: Path, *, create: bool) -> TrustedWorkspaceRoot | None:
    """
    Open the workspace root through pinned parent descriptors.
    """

    absolute_path = path.absolute()

    try:
        root_fd = os.open(absolute_path.anchor, OPEN_DIRECTORY_FLAGS)
    except OSError as exc:
        raise AgentWorkspaceError("Workspace root cannot be opened") from exc

    current_fd = root_fd

    try:
        for index, path_part in enumerate(absolute_path.parts[1:], start=1):
            is_workspace_root = index == len(absolute_path.parts) - 1
            next_fd = _open_workspace_root_component(
                current_fd,
                path_part,
                create=create,
                is_workspace_root=is_workspace_root,
            )

            if next_fd is None:
                return None

            if current_fd != root_fd:
                os.close(current_fd)

            current_fd = next_fd

        _assert_trusted_workspace_root(current_fd)

        try:
            pinned_root_fd = os.dup(current_fd)
        except OSError as exc:
            raise AgentWorkspaceError("Workspace root cannot be pinned") from exc

        return TrustedWorkspaceRoot(
            path=absolute_path,
            fd=pinned_root_fd,
        )
    finally:
        os.close(current_fd)

        if current_fd != root_fd:
            os.close(root_fd)


def _open_workspace_root_component(
    parent_fd: int,
    path_part: str,
    *,
    create: bool,
    is_workspace_root: bool,
) -> int | None:
    """
    Open one workspace-root component without following symlinks.
    """

    try:
        return os.open(path_part, OPEN_DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            return None

        try:
            os.mkdir(path_part, PRIVATE_DIRECTORY_MODE, dir_fd=parent_fd)
        except OSError as exc:
            if not isinstance(exc, FileExistsError):
                raise AgentWorkspaceError("Workspace root cannot be created") from exc

        return _open_workspace_root_component(
            parent_fd,
            path_part,
            create=False,
            is_workspace_root=is_workspace_root,
        )
    except NotADirectoryError as exc:
        raise _workspace_root_component_error(is_workspace_root, symlink=False) from exc
    except OSError as exc:
        if exc.errno == getattr(os, "ELOOP", 62):
            raise _workspace_root_component_error(is_workspace_root, symlink=True) from exc

        raise AgentWorkspaceError("Workspace root cannot be opened") from exc


def _workspace_root_component_error(
    is_workspace_root: bool,
    *,
    symlink: bool,
) -> AgentWorkspaceError:
    """
    Build a deterministic workspace-root component error.
    """

    if symlink:
        if is_workspace_root:
            return AgentWorkspaceError("Workspace root must not be a symlink")

        return AgentWorkspaceError("Workspace root ancestor must not be a symlink")

    if is_workspace_root:
        return AgentWorkspaceError("Workspace root must be a directory")

    return AgentWorkspaceError("Workspace root ancestor must be a directory")


def _assert_trusted_workspace_root(root_fd: int) -> None:
    """
    Verify the pinned workspace root is owned by the orchestration process.
    """

    try:
        root_stat = os.fstat(root_fd)
    except OSError as exc:
        raise AgentWorkspaceError("Workspace root cannot be inspected") from exc

    if not stat.S_ISDIR(root_stat.st_mode):
        raise AgentWorkspaceError("Workspace root must be a directory")

    if root_stat.st_uid != os.geteuid():
        raise AgentWorkspaceError("Workspace root must be owned by the orchestration process")

    if root_stat.st_mode & UNTRUSTED_WRITE_BITS:
        raise AgentWorkspaceError("Workspace root must not be writable by group or other users")


def _create_workspace_directory(root_fd: int, workspace_name: str) -> FilesystemIdentity:
    """
    Create a fresh per-run workspace under a pinned root descriptor.
    """

    try:
        os.mkdir(workspace_name, PRIVATE_DIRECTORY_MODE, dir_fd=root_fd)
    except FileExistsError as exc:
        raise AgentWorkspaceError("Workspace already exists") from exc
    except OSError as exc:
        raise AgentWorkspaceError("Workspace cannot be created") from exc

    try:
        workspace_stat = os.stat(
            workspace_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise WorkspaceIdentityError("Workspace identity cannot be captured") from exc

    if not stat.S_ISDIR(workspace_stat.st_mode):
        raise WorkspaceIdentityError("Workspace identity cannot be captured")

    return _filesystem_identity(workspace_stat)


def _create_workspace_subdirectory(workspace_fd: int, directory_name: str) -> None:
    """
    Create a required workspace subdirectory under a pinned workspace descriptor.
    """

    try:
        os.mkdir(directory_name, dir_fd=workspace_fd)
    except OSError as exc:
        raise AgentWorkspaceError("Workspace subdirectory cannot be created") from exc


def _open_workspace_directory(
    root_fd: int,
    workspace_name: str,
    *,
    expected_identity: FilesystemIdentity | None = None,
) -> _OpenDirectory:
    """
    Open a per-run workspace directory without following leaf symlinks.
    """

    try:
        workspace_fd = os.open(workspace_name, OPEN_DIRECTORY_FLAGS, dir_fd=root_fd)
    except OSError as exc:
        if expected_identity is not None:
            _raise_workspace_identity_error_if_mismatched(
                root_fd,
                workspace_name,
                expected_identity,
            )

        raise AgentWorkspaceError("Workspace cannot be opened") from exc

    return _OpenDirectory(workspace_fd)


def _assert_workspace_identity(
    root_fd: int,
    workspace_fd: int,
    workspace_name: str,
    *,
    expected_identity: FilesystemIdentity | None = None,
) -> None:
    """
    Verify the canonical workspace entry still identifies the pinned directory.
    """

    try:
        canonical_workspace_stat = os.stat(
            workspace_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise WorkspaceIdentityError("Workspace identity cannot be verified") from exc
    except OSError as exc:
        raise WorkspaceIdentityError("Workspace identity cannot be verified") from exc

    if not stat.S_ISDIR(canonical_workspace_stat.st_mode):
        raise WorkspaceIdentityError("Workspace identity cannot be verified")

    try:
        pinned_workspace_stat = os.fstat(workspace_fd)
    except OSError as exc:
        raise WorkspaceIdentityError("Workspace identity cannot be verified") from exc

    pinned_workspace_identity = _filesystem_identity(pinned_workspace_stat)
    canonical_workspace_identity = _filesystem_identity(canonical_workspace_stat)

    if canonical_workspace_identity != pinned_workspace_identity:
        raise WorkspaceIdentityError("Workspace identity changed")

    if expected_identity is not None and pinned_workspace_identity != expected_identity:
        raise WorkspaceIdentityError("Workspace identity changed")


def _raise_workspace_identity_error_if_mismatched(
    root_fd: int,
    workspace_name: str,
    expected_identity: FilesystemIdentity,
) -> None:
    """
    Raise an identity error if a canonical workspace entry no longer matches.
    """

    try:
        current_workspace_stat = os.stat(
            workspace_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise WorkspaceIdentityError("Workspace identity cannot be verified") from exc
    except OSError:
        return

    if not stat.S_ISDIR(current_workspace_stat.st_mode):
        raise WorkspaceIdentityError("Workspace identity cannot be verified")

    if _filesystem_identity(current_workspace_stat) != expected_identity:
        raise WorkspaceIdentityError("Workspace identity changed")


def _assert_workspace_root_identity(path: Path, root_fd: int) -> None:
    """
    Verify the canonical workspace root path still identifies the pinned root.
    """

    try:
        canonical_root_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise WorkspaceIdentityError("Workspace root identity cannot be verified") from exc
    except OSError as exc:
        raise WorkspaceIdentityError("Workspace root identity cannot be verified") from exc

    if not stat.S_ISDIR(canonical_root_stat.st_mode):
        raise WorkspaceIdentityError("Workspace root identity cannot be verified")

    try:
        pinned_root_identity = _filesystem_identity(os.fstat(root_fd))
    except OSError as exc:
        raise WorkspaceIdentityError("Workspace root identity cannot be verified") from exc

    canonical_root_identity = _filesystem_identity(canonical_root_stat)

    if canonical_root_identity != pinned_root_identity:
        raise WorkspaceIdentityError("Workspace root identity changed")


def _filesystem_identity(stat_result: os.stat_result) -> FilesystemIdentity:
    """
    Extract a stable filesystem identity from a stat result.
    """

    return FilesystemIdentity(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
    )


def _assert_bootstrap_failure_preserved_identity(
    trusted_workspace_root: TrustedWorkspaceRoot,
    workspace_fd: int,
    workspace_name: str,
    *,
    expected_identity: FilesystemIdentity,
    original_exception: Exception,
) -> None:
    """
    Verify bootstrap failures did not hide workspace identity loss.
    """

    try:
        _assert_workspace_identity(
            trusted_workspace_root.fd,
            workspace_fd,
            workspace_name,
            expected_identity=expected_identity,
        )
        _assert_workspace_root_identity(
            trusted_workspace_root.path,
            trusted_workspace_root.fd,
        )
    except WorkspaceIdentityError as exc:
        raise exc from original_exception


def _open_context_parent_directory(parent_fd: int, directory_part: str) -> int:
    """
    Open a context destination parent directory without following symlinks.
    """

    try:
        return os.open(directory_part, OPEN_DIRECTORY_FLAGS, dir_fd=parent_fd)
    except NotADirectoryError as exc:
        raise AgentWorkspaceError("Context destination parent must be a directory") from exc
    except OSError as exc:
        if exc.errno == getattr(os, "ELOOP", 62):
            raise AgentWorkspaceError("Context destination parent must not be a symlink") from exc

        raise AgentWorkspaceError("Context destination parent cannot be opened") from exc


def _remove_workspace_directory(root_fd: int, workspace_name: str) -> None:
    """
    Remove a per-run workspace under a pinned root descriptor.
    """

    try:
        workspace_stat = os.stat(
            workspace_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise WorkspaceCleanupError("Workspace cleanup cannot inspect workspace") from exc

    if stat.S_ISLNK(workspace_stat.st_mode):
        _unlink_cleanup_entry(root_fd, workspace_name, missing_ok=True)
        return

    if not stat.S_ISDIR(workspace_stat.st_mode):
        raise WorkspaceCleanupError("Workspace cleanup target must be a directory")

    expected_identity = _filesystem_identity(workspace_stat)

    with _open_cleanup_directory(root_fd, workspace_name) as workspace_fd:
        _assert_cleanup_workspace_identity(
            root_fd,
            workspace_fd,
            workspace_name,
            expected_identity,
        )
        _remove_directory_contents(workspace_fd)
        _assert_cleanup_workspace_identity(
            root_fd,
            workspace_fd,
            workspace_name,
            expected_identity,
        )
        _remove_empty_cleanup_directory(root_fd, workspace_name, missing_ok=False)


def _remove_directory_contents(directory_fd: int) -> None:
    """
    Recursively remove directory contents relative to a pinned descriptor.
    """

    try:
        entry_names = os.listdir(directory_fd)
    except OSError as exc:
        raise WorkspaceCleanupError("Workspace cleanup cannot list directory") from exc

    for entry_name in entry_names:
        _remove_directory_entry(directory_fd, entry_name)


def _remove_directory_entry(parent_fd: int, entry_name: str) -> None:
    """
    Remove one entry from a pinned cleanup directory.
    """

    try:
        entry_stat = os.stat(entry_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise WorkspaceCleanupError("Workspace cleanup cannot inspect entry") from exc

    if stat.S_ISDIR(entry_stat.st_mode):
        _remove_child_directory(parent_fd, entry_name, _filesystem_identity(entry_stat))
        return

    _unlink_cleanup_entry(parent_fd, entry_name, missing_ok=True)


def _remove_child_directory(
    parent_fd: int,
    entry_name: str,
    expected_identity: FilesystemIdentity,
) -> None:
    """
    Remove a child directory only while its canonical entry keeps the pinned identity.
    """

    with _open_cleanup_directory(parent_fd, entry_name) as child_fd:
        _assert_cleanup_workspace_identity(
            parent_fd,
            child_fd,
            entry_name,
            expected_identity,
        )
        _remove_directory_contents(child_fd)
        _assert_cleanup_workspace_identity(
            parent_fd,
            child_fd,
            entry_name,
            expected_identity,
        )
        _remove_empty_cleanup_directory(parent_fd, entry_name, missing_ok=False)


def _open_cleanup_directory(parent_fd: int, entry_name: str) -> _OpenDirectory:
    """
    Pin a cleanup directory entry without following symlinks.
    """

    try:
        directory_fd = os.open(entry_name, OPEN_DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise WorkspaceCleanupError("Workspace cleanup cannot open directory") from exc

    return _OpenDirectory(directory_fd)


def _assert_cleanup_workspace_identity(
    parent_fd: int,
    directory_fd: int,
    entry_name: str,
    expected_identity: FilesystemIdentity,
) -> None:
    """
    Verify a cleanup directory still matches its pinned identity.
    """

    try:
        _assert_workspace_identity(
            parent_fd,
            directory_fd,
            entry_name,
            expected_identity=expected_identity,
        )
    except AgentWorkspaceError as exc:
        raise WorkspaceCleanupError("Workspace cleanup identity changed") from exc


def _unlink_cleanup_entry(parent_fd: int, entry_name: str, *, missing_ok: bool) -> None:
    """
    Unlink a non-directory cleanup entry without following symlinks.
    """

    try:
        os.unlink(entry_name, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        if missing_ok:
            return

        raise WorkspaceCleanupError("Workspace cleanup entry disappeared") from exc
    except OSError as exc:
        raise WorkspaceCleanupError("Workspace cleanup cannot remove entry") from exc


def _remove_empty_cleanup_directory(
    parent_fd: int,
    entry_name: str,
    *,
    missing_ok: bool,
) -> None:
    """
    Remove a pinned cleanup directory's canonical empty entry.
    """

    try:
        os.rmdir(entry_name, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        if missing_ok:
            return

        raise WorkspaceCleanupError("Workspace cleanup directory disappeared") from exc
    except OSError as exc:
        raise WorkspaceCleanupError("Workspace cleanup cannot remove directory") from exc


def _write_context_file_at(directory_fd: int, destination: str, content: bytes) -> None:
    """
    Materialize a validated context snapshot relative to a pinned directory.
    """

    destination_parts = PurePosixPath(destination).parts

    try:
        parent_fd = os.dup(directory_fd)
    except OSError as exc:
        raise AgentWorkspaceError("Context destination cannot be materialized") from exc

    try:
        for directory_part in destination_parts[:-1]:
            try:
                os.mkdir(directory_part, dir_fd=parent_fd)
            except FileExistsError:
                pass
            except OSError as exc:
                raise AgentWorkspaceError("Context destination parent cannot be created") from exc

            next_parent_fd = _open_context_parent_directory(parent_fd, directory_part)
            os.close(parent_fd)
            parent_fd = next_parent_fd

        _write_context_file_name_at(parent_fd, destination_parts[-1], content)
    finally:
        os.close(parent_fd)


def _write_context_file_name_at(directory_fd: int, filename: str, content: bytes) -> None:
    """
    Materialize a validated context snapshot as a new regular file.
    """

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)

    try:
        destination_fd: int | None = os.open(filename, flags, 0o666, dir_fd=directory_fd)
    except FileExistsError as exc:
        raise AgentWorkspaceError("Context destination already exists") from exc
    except OSError as exc:
        raise AgentWorkspaceError("Context destination cannot be opened") from exc

    try:
        try:
            with os.fdopen(destination_fd, "wb") as destination_file:
                destination_fd = None
                destination_file.write(content)
        except OSError as exc:
            raise AgentWorkspaceError("Context destination cannot be written") from exc
    finally:
        if destination_fd is not None:
            try:
                os.close(destination_fd)
            except OSError as exc:
                raise AgentWorkspaceError("Context destination cannot be closed") from exc
