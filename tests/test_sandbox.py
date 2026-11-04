import pytest

from stratarena.sandbox import SandboxedFunction, SandboxError, SandboxLimits, check_source

FAST = SandboxLimits(call_timeout_s=1.0, cpu_seconds=5, memory_mb=256)


def test_calls_return_json_results_and_keep_state() -> None:
    src = "count = 0\ndef f(x):\n    global count\n    count += 1\n    return [x * 2, count]\n"
    with SandboxedFunction(src, "f", FAST) as f:
        assert f(3) == [6, 1]
        assert f(4) == [8, 2]


def test_prints_do_not_break_the_protocol() -> None:
    src = "import os\ndef f():\n    print('noise')\n    os.write(1, b'raw\\n')\n    return 1\n"
    with SandboxedFunction(src, "f", FAST) as f:
        assert f() == 1


def test_exceptions_are_reported() -> None:
    with SandboxedFunction("def f():\n    raise KeyError('boom')\n", "f", FAST) as f:
        with pytest.raises(SandboxError, match="KeyError"):
            f()
        with pytest.raises(SandboxError, match="KeyError"):
            f()


def test_infinite_loop_times_out() -> None:
    limits = SandboxLimits(call_timeout_s=0.3, cpu_seconds=5)
    with SandboxedFunction("def f():\n    while True:\n        pass\n", "f", limits) as f:
        with pytest.raises(SandboxError, match="timed out"):
            f()
        with pytest.raises(SandboxError, match="not running"):
            f()


def test_memory_limit() -> None:
    src = "def f():\n    return len(bytearray(10**9))\n"
    with SandboxedFunction(src, "f", FAST) as f, pytest.raises(SandboxError, match="MemoryError"):
        f()


def test_cannot_start_processes() -> None:
    src = "import subprocess\ndef f():\n    return subprocess.run(['true']).returncode\n"
    with SandboxedFunction(src, "f", FAST) as f, pytest.raises(SandboxError):
        f()


def test_no_third_party_packages() -> None:
    assert check_source("import numpy\ndef f():\n    pass\n", "f", FAST) is not None


def test_non_json_result_is_an_error() -> None:
    src = "def f():\n    return object()\n"
    with SandboxedFunction(src, "f", FAST) as f, pytest.raises(SandboxError, match="JSON"):
        f()


def test_check_source() -> None:
    assert check_source("def f():\n    return 1\n", "f", FAST) is None
    assert "SyntaxError" in (check_source("def f(:\n", "f", FAST) or "")
    assert "KeyError" in (check_source("def g():\n    pass\n", "f", FAST) or "")
    assert "not callable" in (check_source("f = 3\n", "f", FAST) or "")


def test_crash_on_import_is_reported() -> None:
    assert check_source("import os\nos._exit(3)\n", "f", FAST) is not None
