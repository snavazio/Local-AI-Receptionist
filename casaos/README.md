# Receptionist QA on CasaOS

The dashboard has two parts:

1. **Host service** — `web_dashboard.py` (FastAPI) on `localhost:5055`.
   Runs on the host because it needs the GPU, Ollama, and the project venv.
2. **CasaOS tile** — a tiny nginx container on `:5056` that reverse-proxies
   to the host service. Gives you a clickable icon on the CasaOS dashboard.

## 1. Start the host service

One-shot manual run:
```bash
cd ~/projects/receptionist/pipecat-quickstart-phone-bot
.venv/bin/python web_dashboard.py
# open http://<host>:5055/
```

Persistent (recommended) — install the systemd unit:
```bash
sudo cp systemd/receptionist-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now receptionist-dashboard
sudo systemctl status receptionist-dashboard
# logs: journalctl -u receptionist-dashboard -f
```

Verify:
```bash
curl http://localhost:5055/healthz   # → ok
```

## 2. Install the CasaOS app

In the CasaOS web UI:

1. Open the **App Store**.
2. Click the **`+`** icon in the top-right → **Custom Install** → **Import**.
3. Paste the contents of `casaos/docker-compose.yml`.
4. Click **Install**.

Or via CLI on the host:
```bash
casaos-cli app-management install \
  path=/home/snavazio/projects/receptionist/pipecat-quickstart-phone-bot/casaos/docker-compose.yml
```

A **Receptionist QA** tile should appear on your CasaOS dashboard. Click it
and the menu opens in a new tab. Output streams live in the page.

## How it works

```
Browser
   │
   │  http://<casaos-host>:5056/
   ▼
nginx:alpine container  (managed by CasaOS — gives us the tile)
   │
   │  proxy_pass → host.docker.internal:5055
   ▼
web_dashboard.py on the host  (systemd unit, has GPU + Ollama + venv)
   │
   │  subprocess: make qa-smoke / make test / etc.
   ▼
the actual eval pipeline
```

## Security

The host service runs `make` targets over HTTP. Bind it to LAN/localhost
only — do **not** expose port 5055 (or the CasaOS tile port 5056) to the
public internet. The whitelist in `web_dashboard.py` limits which targets
can be invoked, but it's still arbitrary process execution as your user.

## Tweaks

- Different port: edit `DASHBOARD_PORT` in the systemd unit AND
  `proxy_pass` in `casaos/nginx.conf` AND the published port in
  `casaos/docker-compose.yml`. Keep them aligned.
- Add a command: extend the `COMMANDS` dict in `web_dashboard.py`. The new
  button appears on next page load.
- Restrict to localhost only: set `DASHBOARD_HOST=127.0.0.1` in the unit;
  the CasaOS nginx still reaches it via `host.docker.internal`.
