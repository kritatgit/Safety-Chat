# Prompt Safety + GPT Chat

A chat app that classifies user prompts as safe/unsafe, then sends them to an OpenAI model. The safety result can be injected into the system prompt so the model can refuse or reframe unsafe requests.

---

## How it works (workflow)

1. **User** types a prompt in the React UI (`http://localhost:5173`).
2. **Frontend** sends the prompt to the **chat gateway** (`POST /chat` on port 9001).
3. **Chat gateway** (`chat_api.py`):
   - Calls the **safety classifier** (`POST /predict` on port 9000) and gets `label` (safe/unsafe) and probabilities.
   - Builds a system prompt (optional: includes the safety result so the model can refuse unsafe prompts).
   - Calls the **OpenAI API** with the system prompt + user prompt.
   - Returns the assistant reply plus safety label and probabilities to the frontend.
4. **Frontend** shows the reply and the last safety result in the UI.

```
┌─────────────┐     /chat      ┌─────────────┐     /predict     ┌──────────────────┐
│  React UI   │ ──────────────►│  chat_api    │ ────────────────►│  classification  │
│  (port 5173)│                │  (port 9001)│                   │  (port 9000)     │
└─────────────┘                └──────┬──────┘                   └──────────────────┘
                                       │
                                       │  OpenAI API (GPT)
                                       ▼
                                ┌─────────────┐
                                │   OpenAI    │
                                └─────────────┘
```

**Components:**

| Component        | File               | Port | Role                                      |
|-----------------|--------------------|------|-------------------------------------------|
| Safety classifier | `classification.py` | 9000 | `POST /predict` → safe/unsafe + probabilities |
| Chat gateway    | `chat_api.py`      | 9001 | `POST /chat`, `GET /health` → predict + OpenAI → reply |
| Frontend        | `chat-frontend/`   | 5173 | React (Vite) UI; proxies `/chat`, `/health` to 9001 |

---

## Prerequisites

- **Python 3.10+** (for venv and backend)
- **Node.js 18+** and **npm** (for the React frontend)
- **OpenAI API key**
- **Fine-tuned model**: folder with saved tokenizer and model (e.g. `fine_tuned_mobilebert_model_colab/`). Set `MODEL_DIR` in `.env` if you use a different path.

---

## Setup (local)

### 1. Clone and enter the project (if needed)

```powershell
cd path\to\Project-1
```

### 2. Python environment and backend dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root (or set these in your shell). Required for chat:

| Variable           | Description                    | Example                          |
|--------------------|--------------------------------|----------------------------------|
| `OPENAI_API_KEY`   | Your OpenAI API key            | `sk-...`                         |
| `OPENAI_MODEL`     | Model name                     | `gpt-4o-mini`                    |
| `PREDICT_URL`      | Classifier `/predict` URL      | `http://127.0.0.1:9000/predict` |
| `MODEL_DIR`        | Path to fine-tuned model dir   | `./fine_tuned_mobilebert_model_colab` |

Optional (have defaults):

| Variable             | Default                              | Description              |
|----------------------|--------------------------------------|--------------------------|
| `CORS_ORIGINS`       | `http://localhost:5173,http://127.0.0.1:5173` | Allowed frontend origins |
| `CHAT_RATE_LIMIT`    | `10/minute`                          | Rate limit for `/chat`   |
| `PREDICT_RATE_LIMIT` | `30/minute`                          | Rate limit for `/predict`|
| `MAX_PROMPT_LENGTH`  | `2000`                               | Max prompt length        |

Example `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
PREDICT_URL=http://127.0.0.1:9000/predict
MODEL_DIR=./fine_tuned_mobilebert_model_colab
```

### 4. Frontend dependencies

```powershell
cd chat-frontend
npm install
cd ..
```

---

## Run the app (local)

Use **three terminals**. Start the backend first, then the frontend.

### Terminal 1 – Safety classifier (port 9000)

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn classification:app --reload --port 9000
```

### Terminal 2 – Chat gateway (port 9001)

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn chat_api:app --reload --port 9001
```

### Terminal 3 – React UI (port 5173)

```powershell
cd chat-frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

**Check that the gateway is up:**

```powershell
Invoke-RestMethod http://127.0.0.1:9001/health
```

You should see something like `{"ok":true,"predict_url":"...","openai_model":"..."}`.

---

## Quick reference – ports

| Service           | Port | Command                                      |
|-------------------|------|----------------------------------------------|
| Safety classifier | 9000 | `uvicorn classification:app --reload --port 9000` |
| Chat gateway      | 9001 | `uvicorn chat_api:app --reload --port 9001`  |
| React frontend    | 5173 | `cd chat-frontend && npm run dev`            |

The frontend proxies `/chat` and `/health` to the chat gateway at 9001, so you only need to open the UI at 5173.
