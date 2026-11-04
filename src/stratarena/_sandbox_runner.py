"""Child side of the sandbox protocol. Standard library only; run with ``python -I -S``.

Protocol (one JSON object per line): the parent first sends ``{"source", "entrypoint"}``
and gets ``{"ok": true}`` once the source is loaded; then each ``{"args": [...]}`` gets
``{"ok": true, "result": ...}`` or ``{"ok": false, "error": "..."}``.
"""

import json
import os
import sys


def main() -> None:  # pragma: no cover - exercised through subprocesses in the tests
    # Keep the real stdout for the protocol and point fd 1 at stderr, so prints (or even
    # os.write(1, ...)) in the strategy code cannot corrupt the protocol stream.
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj: object) -> None:
        try:
            line = json.dumps(obj)
        except (TypeError, ValueError) as e:
            line = json.dumps({"ok": False, "error": f"result is not JSON: {e}"})
        protocol.write(line + "\n")
        protocol.flush()

    hello = json.loads(sys.stdin.readline())
    namespace: dict[str, object] = {"__name__": "strategy"}
    try:
        exec(compile(hello["source"], "<strategy>", "exec"), namespace)
        fn = namespace[hello["entrypoint"]]
        if not callable(fn):
            raise TypeError(f"{hello['entrypoint']} is not callable")
    except BaseException as e:
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    send({"ok": True})

    for line in sys.stdin:
        args = json.loads(line)["args"]
        try:
            send({"ok": True, "result": fn(*args)})
        except BaseException as e:
            send({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
