#!/usr/bin/env python3
"""Smoke-test the in-container clangd LSP through the project's .nvim/clangd wrapper.

Usage:  python3 tools/lsp_features.py [project_root]   (default: repo root)
Or:     mise run lsp:check

Drives .nvim/clangd (docker exec + --path-mappings) over stdio, opens
src/common/hello.cu, and issues one request per LSP feature, printing FOUND / empty /
ERR. No nvim needed -- it validates the server side of this container setup.
Exit code is non-zero if the client can't even initialize.
"""

import json
import os
import pathlib
import subprocess
import sys
import threading
import time

ROOT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
FILE = os.path.join(ROOT, "src", "common", "hello.cu")
uri = "file://" + FILE
text = pathlib.Path(FILE).read_text()
lines = text.splitlines()


def pos(needle, occurrence=1, offset=0):
    seen = 0
    for idx, line in enumerate(lines):
        col = line.find(needle)
        if col >= 0:
            seen += 1
            if seen == occurrence:
                return {"line": idx, "character": col + offset}
    raise SystemExit(f"needle not found in hello.cu: {needle!r}")


wrapper = os.path.join(ROOT, ".nvim", "clangd")
if not os.access(wrapper, os.X_OK):
    raise SystemExit(f"missing/again non-executable wrapper: {wrapper}")

p = subprocess.Popen(
    [wrapper], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)
responses, lock = {}, threading.Lock()


def send(obj):
    b = json.dumps(obj).encode()
    p.stdin.write(f"Content-Length: {len(b)}\r\n\r\n".encode() + b)
    p.stdin.flush()


def reader():
    buf = b""
    while True:
        c = p.stdout.read(1)
        if not c:
            return
        buf += c
        if buf.endswith(b"\r\n\r\n"):
            hdr = dict(x.split(b": ") for x in buf.split(b"\r\n") if b": " in x)
            body = p.stdout.read(int(hdr[b"Content-Length"]))
            msg = json.loads(body)
            if isinstance(msg.get("id"), int) and "method" not in msg:
                with lock:
                    responses[msg["id"]] = msg
            buf = b""


threading.Thread(target=reader, daemon=True).start()

_id = [10]


def request(method, params, wait=5.0):
    _id[0] += 1
    i = _id[0]
    send({"jsonrpc": "2.0", "id": i, "method": method, "params": params})
    deadline = time.time() + wait
    while time.time() < deadline:
        with lock:
            if i in responses:
                return responses[i]
        time.sleep(0.02)
    return None


CAPS = {
    "textDocument": {
        k: {}
        for k in (
            "hover",
            "definition",
            "references",
            "documentSymbol",
            "documentHighlight",
            "signatureHelp",
            "completion",
            "formatting",
            "rename",
            "foldingRange",
            "selectionRange",
            "inlayHint",
            "codeAction",
            "typeDefinition",
            "implementation",
        )
    }
}
CAPS["textDocument"]["semanticTokens"] = {
    "requests": {"full": True},
    "tokenTypes": [],
    "tokenModifiers": [],
    "formats": ["relative"],
}

init = request(
    "initialize",
    {
        "processId": None,
        "rootUri": "file://" + ROOT,
        "capabilities": CAPS,
        "workspaceFolders": [{"uri": "file://" + ROOT, "name": "cuda"}],
    },
)
if not init or "result" not in init:
    print(
        "FAILED: clangd did not initialize (is the container running? `mise run dc:up`)"
    )
    p.terminate()
    sys.exit(1)
send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
send(
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {
            "textDocument": {
                "uri": uri,
                "languageId": "cuda",
                "version": 1,
                "text": text,
            }
        },
    }
)
time.sleep(6)

kernel = pos("hello_kernel", 2)
kernel_def = pos("hello_kernel", 1)
sync = pos("cudaDeviceSynchronize", 1, 4)
printf_open = pos("printf(", 1, 7)
td = {"textDocument": {"uri": uri}}
nonempty = lambda r: bool(r)


def verdict(r, ok):
    if r is None:
        return "TIMEOUT"
    if "error" in r:
        return "ERR " + str(r["error"].get("message", ""))[:50]
    return "FOUND" if ok(r.get("result")) else "empty"


tests = [
    ("hover", "textDocument/hover", {**td, "position": sync}, nonempty),
    ("definition", "textDocument/definition", {**td, "position": kernel}, nonempty),
    (
        "references",
        "textDocument/references",
        {**td, "position": kernel_def, "context": {"includeDeclaration": True}},
        nonempty,
    ),
    ("documentSymbol", "textDocument/documentSymbol", td, nonempty),
    (
        "documentHighlight",
        "textDocument/documentHighlight",
        {**td, "position": kernel_def},
        nonempty,
    ),
    (
        "signatureHelp",
        "textDocument/signatureHelp",
        {**td, "position": printf_open},
        lambda r: bool(r and r.get("signatures")),
    ),
    (
        "completion",
        "textDocument/completion",
        {**td, "position": sync},
        lambda r: bool(r and (r.get("items") or r)),
    ),
    (
        "semanticTokens",
        "textDocument/semanticTokens/full",
        td,
        lambda r: bool(r and r.get("data")),
    ),
    (
        "formatting",
        "textDocument/formatting",
        {**td, "options": {"tabSize": 4, "insertSpaces": True}},
        lambda r: r is not None,
    ),
    ("foldingRange", "textDocument/foldingRange", td, nonempty),
    (
        "inlayHint",
        "textDocument/inlayHint",
        {
            **td,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": len(lines), "character": 0},
            },
        },
        nonempty,
    ),
    (
        "rename",
        "textDocument/rename",
        {**td, "position": kernel_def, "newName": "greet_kernel"},
        lambda r: bool(r and r.get("changes")),
    ),
]

print(f"clangd LSP feature check ({os.path.basename(ROOT)}/src/common/hello.cu):")
width = max(len(t[0]) for t in tests)
fails = 0
for name, method, params, ok in tests:
    r = request(method, params)
    v = verdict(r, ok)
    if v.startswith("ERR") or v == "TIMEOUT":
        fails += 1
    print(f"  {name:<{width}}  {v}")

send({"jsonrpc": "2.0", "id": 999, "method": "shutdown", "params": None})
time.sleep(0.3)
p.terminate()
sys.exit(1 if fails else 0)
