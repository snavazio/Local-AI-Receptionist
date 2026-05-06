"""Web dashboard — same options as dashboard.sh, but as a browser UI.

Runs on the HOST (not in a container) because the eval pipeline needs the
host's GPU, Ollama, and the project's .venv with CUDA wheels. CasaOS shows
a tile that reverse-proxies to this Flask app (see casaos/docker-compose.yml).

Start manually:
    .venv/bin/python web_dashboard.py
or via systemd:
    systemctl --user enable --now receptionist-dashboard

Then open http://<host>:5055/

Security note: this exposes `make` targets over HTTP. Bind to localhost or
your LAN only — do NOT expose to the public internet.

Built on FastAPI + uvicorn (already in the project's .venv via pipecat).
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("DASHBOARD_PORT", "5055"))
HOST = os.environ.get("DASHBOARD_HOST", "0.0.0.0")

# Whitelist of allowed commands. Anything not in here cannot be run.
COMMANDS = {
    "qa-smoke":      ("Smoke eval (~5 min)",        ["make", "qa-smoke"],     "31 cases, fast iteration"),
    "qa-full":       ("Full eval (~30-50 min)",     ["make", "qa-full"],      "All 356 cases"),
    "qa-judge":      ("Smoke + judge",              ["make", "qa-judge"],     "Adds LLM-as-judge scoring"),
    "qa-multi":      ("Multi-model bake-off",       ["make", "qa-multi"],     "Default: qwen2.5:14b vs 7b"),
    "watch":         ("Regression watcher",         ["make", "watch"],        "Eval + diff vs baseline"),
    "test":          ("Unit tests",                 ["make", "test"],         "~2 sec, no GPU"),
    "test-verbose":  ("Unit tests (verbose)",       ["make", "test-verbose"], "Per-test output"),
    "lint":          ("Lint",                       ["bash", "-c", "make lint && make lint-yaml"], "AST + cases.yaml"),
    "status":        ("Status snapshot",            ["make", "status"],       "Git/Ollama/GPU/baseline"),
    "trend":         ("Trend (last 10 runs)",       ["make", "trend"],        "ASCII sparklines"),
    "last-report":   ("Last QA report",             ["make", "last-report"],  "Most recent qa_*.md"),
    "clean":         ("Clean ephemeral reports",    ["make", "clean"],        "Keeps baseline + history"),
    "help":          ("Show make help",             ["make", "help"],         "Full target list"),
}

# In-memory job registry.
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _run_job(job_id: str, argv: list[str]) -> None:
    job = _JOBS[job_id]
    try:
        proc = subprocess.Popen(
            argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        job["pid"] = proc.pid
        assert proc.stdout is not None
        for line in proc.stdout:
            with _LOCK:
                job["lines"].append(line.rstrip("\n"))
        proc.wait()
        job["rc"] = proc.returncode
    except Exception as e:
        with _LOCK:
            job["lines"].append(f"[dashboard] ERROR: {e}")
            job["rc"] = -1
    finally:
        job["done"] = True
        job["finished"] = time.time()


app = FastAPI(title="Receptionist QA Dashboard")


INDEX_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Receptionist QA Dashboard</title>
<style>
  body { font: 14px/1.4 system-ui, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 16px; background: #0e1116; color: #d6dae0; }
  h1 { font-size: 18px; border-bottom: 1px solid #2a2f37; padding-bottom: 8px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(220px,1fr)); gap: 8px; margin-bottom: 16px; }
  button { background: #1c222b; color: #d6dae0; border: 1px solid #2a3340; border-radius: 6px; padding: 10px 12px; cursor: pointer; text-align: left; font: inherit; }
  button:hover { background: #243040; border-color: #3a4a60; }
  button.running { background: #3a2a18; border-color: #6a4a20; }
  button .desc { display: block; color: #7a8290; font-size: 11px; margin-top: 3px; }
  pre { background: #06080b; color: #b8c0cc; padding: 12px; border-radius: 6px; min-height: 240px; max-height: 60vh; overflow: auto; white-space: pre-wrap; word-break: break-word; }
  .bar { display: flex; gap: 8px; align-items: center; margin: 8px 0; font-size: 12px; color: #7a8290; }
  .bar .ok { color: #5fbf5f; } .bar .fail { color: #ff7070; } .bar .run { color: #f0a050; }
</style>
</head><body>
<h1>Receptionist QA — Dashboard</h1>
<div class="grid" id="grid"></div>
<div class="bar">
  <span id="status">idle</span>
  <span style="flex:1"></span>
  <button id="stop" onclick="stopJob()" style="display:none">Stop</button>
  <button onclick="clearLog()">Clear log</button>
</div>
<pre id="log">Pick a command above. Output streams here.</pre>
<script>
const COMMANDS = __COMMANDS__;
let currentJob = null, lastLen = 0;

function render() {
  const g = document.getElementById("grid");
  g.innerHTML = "";
  for (const [id, c] of Object.entries(COMMANDS)) {
    const b = document.createElement("button");
    b.id = "btn-" + id;
    b.onclick = () => run(id);
    b.appendChild(document.createTextNode(c.label));
    const d = document.createElement("span");
    d.className = "desc"; d.textContent = c.desc;
    b.appendChild(d);
    g.appendChild(b);
  }
}

async function run(id) {
  if (currentJob) { alert("A job is already running. Stop it first."); return; }
  document.getElementById("log").textContent = "";
  lastLen = 0;
  const r = await fetch("/run/" + id, {method: "POST"});
  if (!r.ok) { alert("Failed to start: " + await r.text()); return; }
  const j = await r.json();
  currentJob = j.job_id;
  document.getElementById("btn-" + id).classList.add("running");
  document.getElementById("status").innerHTML = '<span class="run">▶ running ' + id + '</span>';
  document.getElementById("stop").style.display = "";
  poll();
}

async function poll() {
  if (!currentJob) return;
  const r = await fetch("/log/" + currentJob + "?since=" + lastLen);
  const j = await r.json();
  if (j.lines && j.lines.length) {
    const log = document.getElementById("log");
    log.textContent += j.lines.join("\\n") + "\\n";
    log.scrollTop = log.scrollHeight;
    lastLen = j.next;
  }
  if (j.done) {
    const stat = document.getElementById("status");
    stat.innerHTML = j.rc === 0
      ? '<span class="ok">✓ exit 0</span>'
      : '<span class="fail">✗ exit ' + j.rc + '</span>';
    document.querySelectorAll("button.running").forEach(b => b.classList.remove("running"));
    document.getElementById("stop").style.display = "none";
    currentJob = null;
    return;
  }
  setTimeout(poll, 600);
}

async function stopJob() {
  if (!currentJob) return;
  await fetch("/stop/" + currentJob, {method: "POST"});
}

function clearLog() {
  document.getElementById("log").textContent = "";
  lastLen = 0;
}

render();
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    cmds = {k: {"label": v[0], "desc": v[2]} for k, v in COMMANDS.items()}
    return INDEX_HTML.replace("__COMMANDS__", json.dumps(cmds))


@app.post("/run/{cmd_id}")
def run_cmd(cmd_id: str):
    if cmd_id not in COMMANDS:
        raise HTTPException(404, "unknown command")
    for j in _JOBS.values():
        if not j.get("done"):
            raise HTTPException(409, "a job is already running")
    _, argv, _ = COMMANDS[cmd_id]
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "cmd": argv, "cmd_id": cmd_id, "started": time.time(),
        "lines": [f"$ {' '.join(shlex.quote(a) for a in argv)}"],
        "done": False, "rc": None, "pid": None,
    }
    threading.Thread(target=_run_job, args=(job_id, argv), daemon=True).start()
    return {"job_id": job_id}


@app.get("/log/{job_id}")
def log_(job_id: str, since: int = 0):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    with _LOCK:
        lines = job["lines"][since:]
        nxt = len(job["lines"])
    return {"lines": lines, "next": nxt, "done": job["done"], "rc": job["rc"]}


@app.post("/stop/{job_id}")
def stop_(job_id: str):
    job = _JOBS.get(job_id)
    if not job or job.get("done"):
        raise HTTPException(404, "not running")
    pid = job.get("pid")
    if pid:
        try:
            # Kill the whole process group so child make/python procs die too.
            os.killpg(pid, signal.SIGTERM)
            return PlainTextResponse("ok")
        except ProcessLookupError:
            return PlainTextResponse("gone")
    raise HTTPException(409, "no pid yet")


@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")


if __name__ == "__main__":
    import uvicorn
    print(f"[dashboard] http://{HOST}:{PORT}/   project={ROOT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
