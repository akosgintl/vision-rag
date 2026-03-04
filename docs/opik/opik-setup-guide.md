# Setting Up Opik: Workspace, Project & API Key

Opik offers **two deployment modes**. Some setup steps overlap, others are completely different. This guide covers both side by side so you always know what applies to you.

```
┌──────────────────────────────────────────────────────────────┐
│                   Which Opik are you using?                  │
│                                                              │
│   ☁️  Opik Cloud  ──→  Hosted by Comet at comet.com/opik     │
│       • Requires: API key, workspace name                    │
│       • Includes: user management, billing, support          │
│                                                              │
│   🖥️  Self-Hosted  ──→  Runs on your own machine / cluster   │
│       • Requires: Docker (local) or Kubernetes (production)  │
│       • No API key needed                                    │
│       • No user management                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Part 1 — Opik Cloud (Hosted by Comet)

### Step 1: Create a Free Account (= Your Workspace)

1. Go to **[comet.com/opik](https://comet.com/opik)**.
2. Click **Sign up** (you can use GitHub, Google, or email).
3. Once signed in, you land in your **workspace**.

Your **workspace name** is visible in the breadcrumb at the top of the page:

```
opik  ›  your-workspace-name  ›  Projects  ›  ...
```

A workspace is your top-level container — think of it as your "organization" or "team account." All projects, traces, and prompts live inside it.

### Step 2: Find or Create Your API Key

The API key authenticates your local SDK against the cloud. It starts with `opik_...`.

**Where to find it:**

1. Click your **profile icon** (top-right corner of the Opik UI).
2. Go to **Account Settings** (or **API Keys**).
3. Copy your existing key, or click **Create new API key**.

> **Tip:** Store the key in an environment variable (`OPIK_API_KEY`) so it never ends up in your code.

### Step 3: Create a Project

A **project** groups related traces together (e.g. "chatbot-prod", "rag-pipeline-dev").

**Option A — Via the UI:**

1. In the sidebar, click **Projects**.
2. Click **+ New Project**.
3. Give it a name (e.g. `my-extraction-pipeline`).

**Option B — Automatically from code:**

If you specify a `project_name` that doesn't exist yet, Opik creates it for you on the first trace. No manual step needed.

### Step 4: Configure the Python SDK

Install and configure:

```bash
pip install opik
opik configure
```

The interactive prompt will ask for:

| Prompt                  | What to enter                        |
|-------------------------|--------------------------------------|
| Opik deployment         | **Cloud**                            |
| API key                 | Your `opik_...` key                  |
| Workspace               | Your workspace name                  |

This writes a config file to `~/.opik.config`. Alternatively, configure in code:

```python
import opik

opik.configure(
    api_key="opik_your_key_here",
    workspace="your-workspace-name",
)
```

Or purely via environment variables (great for CI/CD and Docker):

```bash
export OPIK_API_KEY="opik_your_key_here"
export OPIK_WORKSPACE="your-workspace-name"
export OPIK_PROJECT_NAME="my-extraction-pipeline"
# URL defaults to https://www.comet.com/opik/api — no override needed
```

### Step 5: Verify the Connection

```python
from opik import track

@track
def hello(name: str) -> str:
    return f"Hello, {name}!"

hello("world")
```

Open the Opik Cloud dashboard → your project → you should see a trace.

---

## Part 2 — Self-Hosted / Local (Docker Compose)

### Step 1: Prerequisites

You need **Git** and **Docker** (with Docker Compose) installed.

On Mac/Windows, Docker Desktop includes Docker Compose. On Linux, install them separately if needed.

### Step 2: Install & Start Opik

```bash
# Clone the repository
git clone https://github.com/comet-ml/opik.git

# Enter the folder
cd opik

# Start Opik (Linux / Mac)
./opik.sh

# Start Opik (Windows PowerShell)
.\opik.ps1
```

That's it. Opik is now running at:

```
http://localhost:5173
```

> **Data persistence:** All data is stored in `~/opik` via Docker volumes. You can stop and restart without losing anything.

### Step 3: There Is No API Key

This is the biggest difference from the cloud version:

```
┌─────────────────────────────────────────────────────────────┐
│  🔑  Self-Hosted Opik does NOT use API keys.                │
│      There is no authentication layer.                      │
│      Anyone who can reach the URL can read/write data.      │
│      (This is fine for local dev; for production, consider  │
│       Opik Cloud or Kubernetes with network policies.)      │
└─────────────────────────────────────────────────────────────┘
```

### Step 4: There Is No "Workspace" to Create

On the self-hosted version there is no multi-user workspace concept. You go directly to projects.

### Step 5: Create a Project

Same as cloud — you can create projects in the UI or let Opik auto-create them when you log your first trace with a given `project_name`.

### Step 6: Configure the Python SDK

```bash
pip install opik
opik configure --use_local
```

Or in Python:

```python
import opik

opik.configure(use_local=True)
```

Or via environment variables:

```bash
export OPIK_URL_OVERRIDE="http://localhost:5173/api"
export OPIK_PROJECT_NAME="my-extraction-pipeline"
# No OPIK_API_KEY needed
# No OPIK_WORKSPACE needed
```

### Step 7: Verify

```python
from opik import track

@track
def hello(name: str) -> str:
    return f"Hello, {name}!"

hello("world")
```

Open `http://localhost:5173` → your project → you should see a trace.

### Stopping & Restarting

```bash
# Stop
./opik.sh --stop

# Restart / Upgrade (data is preserved)
./opik.sh
```

---

## Side-by-Side Comparison

| Concept              | ☁️ Opik Cloud                                  | 🖥️ Self-Hosted (Local)                         |
|----------------------|------------------------------------------------|------------------------------------------------|
| **Sign up**          | Create account at comet.com/opik               | Clone repo + `./opik.sh`                       |
| **Workspace**        | Created on sign-up; name in breadcrumb         | Does not exist (no multi-user support)         |
| **API Key**          | Required (`opik_...`), found in Account Settings | **Not needed** — no auth layer                |
| **Project**          | Create in UI or auto-created from code         | Same                                           |
| **SDK config**       | `opik configure` (prompts for key + workspace) | `opik configure --use_local`                   |
| **Dashboard URL**    | `https://www.comet.com/opik/...`               | `http://localhost:5173`                        |
| **Data storage**     | Comet's cloud infrastructure                   | Local Docker volumes (`~/opik`)                |
| **User management**  | Yes (teams, roles, billing)                    | No                                             |
| **Production ready** | Yes                                            | No (use Kubernetes Helm chart for production)  |

---

## Environment Variables Reference

These work for **both** deployment modes (just leave cloud-only ones unset for self-hosted):

| Variable              | Required For   | Description                                                  |
|-----------------------|----------------|--------------------------------------------------------------|
| `OPIK_API_KEY`        | Cloud only     | Your API key (starts with `opik_...`).                       |
| `OPIK_WORKSPACE`      | Cloud only     | Your workspace name.                                         |
| `OPIK_URL_OVERRIDE`   | Self-hosted    | URL of your Opik instance (e.g. `http://localhost:5173/api`).|
| `OPIK_PROJECT_NAME`   | Both (optional)| Default project to log traces to.                            |
| `OPIK_TRACK_DISABLE`  | Both (optional)| Set to `true` to disable all tracing (useful in tests).      |

---

## Configuration Precedence

Both the Python and TypeScript SDKs resolve settings in this order (first match wins):

```
1. Constructor arguments     ──→  opik.configure(api_key="...", ...)
2. Environment variables     ──→  OPIK_API_KEY, OPIK_WORKSPACE, ...
3. Config file               ──→  ~/.opik.config
4. Defaults                  ──→  URL = Opik Cloud, project = "Default Project"
```

This means you can have a `~/.opik.config` for day-to-day work and override it with environment variables in CI/CD without changing any code.

---

## Quick Decision Guide

```
Do you want to get started in 2 minutes with zero infrastructure?
  └─ YES  ──→  Use Opik Cloud (free tier available)

Do you need data to stay on your machine / network?
  └─ YES  ──→  Use Self-Hosted

Do you need user management, roles, and billing?
  └─ YES  ──→  Use Opik Cloud (or Comet Enterprise)

Are you deploying to production at scale?
  └─ YES  ──→  Use Opik Cloud or Self-Hosted on Kubernetes
  └─ NO   ──→  Docker Compose local install is fine
```
