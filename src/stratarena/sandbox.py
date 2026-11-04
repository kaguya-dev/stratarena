"""Run untrusted strategy code in a separate, resource-limited Python process.

The arena's simulators stay in the parent process; the child only computes decisions.
Each call sends JSON arguments to the child and waits for a JSON result, with a
wall-clock timeout per call. The child runs ``python -I -S`` (isolated mode, no
site-packages: standard library only) with a clean environment, a temporary working
directory and POSIX resource limits on CPU time, address space, file size, open files
and process creation.

This contains accidents (infinite loops, memory blow-ups, crashes, stray prints) and
keeps a strategy from tampering with the simulator or its score. It is **not** a security
boundary against deliberately malicious code: the child can still use the network and
read files the user can read. Only run code you are willing to run on your machine.
"""

import contextlib
import json
import os
import select
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

_RUNNER = Path(__file__).with_name("_sandbox_runner.py")


@dataclass(frozen=True)
class SandboxLimits:
    call_timeout_s: float = 2.0
    """Wall-clock limit for one call (and five times this for start-up)."""
    cpu_seconds: int = 30
    """Total CPU time of the child process."""
    memory_mb: int = 256
    """Address-space limit of the child process."""


class SandboxError(RuntimeError):
    """The sandboxed code failed: it raised, timed out, crashed or broke the protocol."""


def _limit_resources(limits: SandboxLimits) -> None:  # pragma: no cover - runs in the child
    import resource

    mem = limits.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setsid()


class SandboxedFunction:
    """A function defined by untrusted source, callable from the parent process.

    Use as a context manager; the child process lives until ``__exit__``. Module-level
    state in the source persists between calls of the same instance.
    """

    def __init__(self, source: str, entrypoint: str, limits: SandboxLimits | None = None) -> None:
        self.source = source
        self.entrypoint = entrypoint
        self.limits = limits or SandboxLimits()
        self._proc: subprocess.Popen[bytes] | None = None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._buffer = b""

    def __enter__(self) -> Self:
        if sys.platform == "win32":  # pragma: no cover
            raise SandboxError("the sandbox needs a POSIX system")
        self._tmp = tempfile.TemporaryDirectory(prefix="stratarena-sandbox-")
        stderr = open(Path(self._tmp.name) / "stderr.txt", "wb")
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-I", "-S", str(_RUNNER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                cwd=self._tmp.name,
                env={},
                preexec_fn=lambda: _limit_resources(self.limits),
            )
        finally:
            stderr.close()
        try:
            self._request({"source": self.source, "entrypoint": self.entrypoint}, startup=True)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()
            self._proc.wait()
            for stream in (self._proc.stdin, self._proc.stdout):
                if stream is not None:
                    stream.close()
            self._proc = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __call__(self, *args: Any) -> Any:
        """Call the entrypoint with JSON-serializable ``args`` and return its JSON result."""
        return self._request({"args": list(args)})

    def _request(self, message: dict[str, Any], startup: bool = False) -> Any:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise SandboxError("sandbox is not running")
        try:
            proc.stdin.write(json.dumps(message).encode() + b"\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise SandboxError(f"sandbox process died{self._stderr_tail()}") from e
        timeout = self.limits.call_timeout_s * (5 if startup else 1)
        line = self._readline(timeout)
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as e:
            raise SandboxError(f"protocol error: {line[:200]!r}") from e
        if not isinstance(reply, dict) or not reply.get("ok"):
            error = reply.get("error", "unknown error") if isinstance(reply, dict) else reply
            raise SandboxError(str(error))
        return reply.get("result")

    def _readline(self, timeout: float) -> bytes:
        assert self._proc is not None and self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise SandboxError(f"timed out after {timeout:.1f}s")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise SandboxError(f"sandbox process exited{self._stderr_tail()}")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line

    def _stderr_tail(self) -> str:
        if self._tmp is None:
            return ""
        path = Path(self._tmp.name) / "stderr.txt"
        tail = path.read_bytes()[-300:].decode(errors="replace").strip() if path.exists() else ""
        return f": {tail}" if tail else ""


def check_source(source: str, entrypoint: str, limits: SandboxLimits | None = None) -> str | None:
    """Return ``None`` if the source compiles and defines ``entrypoint``, else the error."""
    try:
        with SandboxedFunction(source, entrypoint, limits):
            return None
    except SandboxError as e:
        return str(e)
