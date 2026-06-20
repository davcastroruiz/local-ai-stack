# Local AI Stack

A self-hosted AI stack running locally with Docker Compose.

This repository contains everything needed to run:

* Ollama (local LLM inference)
* Open WebUI (chat interface)
* SearXNG (web search)
* ComfyUI (image generation workflows)
* n8n (automation and agents)

---

## Architecture

```text
┌──────────────┐
│ Open WebUI   │
└──────┬───────┘
       │
       ├──► Ollama
       │
       ├──► SearXNG
       │
       ├──► ComfyUI
       │
       └──► n8n

```

---

## Services

| Service    | URL                    | Purpose              |
| ---------- | ---------------------- | -------------------- |
| Open WebUI | http://localhost:3000  | Chat interface       |
| Ollama     | http://localhost:11434 | Local LLM backend    |
| SearXNG    | http://localhost:8080  | Private web search   |
| ComfyUI    | http://localhost:8188  | Image generation     |
| n8n        | http://localhost:5678  | Automations & agents |

---

## Requirements

### Hardware

Recommended:

* NVIDIA GPU
* 16 GB+ VRAM
* 64 GB+ RAM

Current development machine:

* AMD Ryzen 7 8700G
* NVIDIA RTX 5070 eGPU
* 96 GB RAM

---

## Installation

Clone repository:

```bash
git clone https://github.com/davcastroruiz/local-ai-stack.git
cd local-ai-stack
```

Start stack:

```bash
docker compose up -d
```

Verify:

```bash
docker ps
```

---

## Updating

Update all containers:

```bash
docker compose pull
docker compose down
docker compose up -d
```

Update a single service:

```bash
docker compose pull open-webui
docker compose up -d open-webui
```

Examples:

```bash
docker compose pull ollama
docker compose up -d ollama

docker compose pull searxng
docker compose up -d searxng

docker compose pull comfyui
docker compose up -d comfyui

docker compose pull n8n
docker compose up -d n8n
```

---

## Watchtower (Optional)

Automatic container updates.

Example service:

```yaml
watchtower:
  image: containrrr/watchtower
  container_name: watchtower
  restart: unless-stopped
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  command: --cleanup --interval 86400
```

Start:

```bash
docker compose up -d watchtower
```

Logs:

```bash
docker logs -f watchtower
```

Force update check:

```bash
docker exec watchtower /watchtower --run-once
```

---

## SearXNG Configuration

Open WebUI search integration requires JSON output.

Enter container:

```bash
docker exec -it searxng sh
```

Edit:

```bash
/etc/searxng/settings.yml
```

Required configuration:

```yaml
search:
  formats:
    - html
    - json
```

Restart:

```bash
docker restart searxng
```

Validation:

```bash
curl "http://localhost:8080/search?q=test&format=json"
```

Expected result:

* JSON response
* Not HTML

---

## Logs

```bash
docker logs -f ollama
docker logs -f open-webui
docker logs -f searxng
docker logs -f comfyui
docker logs -f n8n
```

---

## Restart Services

```bash
docker restart ollama
docker restart open-webui
docker restart searxng
docker restart comfyui
docker restart n8n
```

---

## Backups

Important data:

### Open WebUI

```text
webui volume
```

### Ollama

```text
ollama volume
```

### n8n

```text
./n8n_data
```

### Compose Configuration

```bash
docker compose config > compose-resolved.yml
```

---

## Migration to a New Machine

1. Install Docker Desktop
2. Copy repository
3. Restore volumes
4. Restore `n8n_data`
5. Run:

```bash
docker compose up -d
```

6. Verify services

---

## Troubleshooting

### Open WebUI cannot search the web

Verify:

```bash
curl "http://localhost:8080/search?q=test&format=json"
```

If HTML is returned:

```yaml
search:
  formats:
    - html
    - json
```

Restart SearXNG.

---

### Verify running containers

```bash
docker ps
```

Expected:

```text
ollama
open-webui
searxng
comfyui
n8n
```

---

## Future Improvements

* HTTPS reverse proxy
* Authentication hardening
* Automatic backups
* Monitoring
* GPU metrics
* Multi-user support
* Agent workflows through n8n
* ComfyUI API integration with Open WebUI

---

## License

MIT License
