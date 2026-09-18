"""
A lock file that stops two copies of the app opening one library.

Chroma's on-disk store is not safe for two processes writing at once. The lock
records the owning process ID; a lock whose process no longer exists (the app
crashed or was killed) is stale and is taken over, so a crash can never lock a
user out of their own library. The same process may open a library again --
several browser tabs share one Streamlit process, which is fine.
"""

import os
from pathlib import Path
from typing import Optional


class LibraryLocked(RuntimeError):
    """The library is open in another running copy of the app."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by someone else.
        return True
    except OSError:
        return False
    return True


def lock_owner(path: Path) -> Optional[int]:
    """PID recorded in the lock file, or None if absent/unreadable."""
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def held_by_other_process(path: Path) -> bool:
    """True when a live process other than this one holds the lock."""
    pid = lock_owner(path)
    return pid is not None and pid != os.getpid() and _pid_alive(pid)


class LibraryLock:
    def __init__(self, path: Path):
        self.path = Path(path)

    def acquire(self) -> None:
        if held_by_other_process(self.path):
            raise LibraryLocked(
                f"This library is open in another running copy of the app "
                f"(process {lock_owner(self.path)}). Close it there first."
            )
        self.path.write_text(str(os.getpid()))

    def release(self) -> None:
        if lock_owner(self.path) == os.getpid():
            try:
                self.path.unlink()
            except OSError:
                pass
