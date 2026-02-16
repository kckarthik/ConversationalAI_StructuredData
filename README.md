# 🧠 Analytics Agent

Production-ready conversational analytics tool. Upload CSVs, ask questions in natural language, get charts + insights powered by a 7-tool AI agent.

```
CSV Upload → SQLite → Intent Planner → 7 Tools → Charts + Insights
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐
│  Browser UI  │────▶│  Flask API   │────▶│  Agent Orchestrator          │
│  HTML/CSS/JS │◀────│  Gunicorn    │◀────│                              │
└──────────────┘     └──────┬───────┘     │  Intent Classifier (rules)   │
                            │             │  ├─ run_sql                   │
                     ┌──────▼───────┐     │  ├─ create_chart (7 types)   │
                     │   SQLite DB  │     │  ├─ profile_table            │
                     │  (from CSVs) │     │  ├─ detect_anomalies         │
                     └──────────────┘     │  ├─ compare_groups           │
                                          │  ├─ trend_analysis           │
                     ┌──────────────┐     │  └─ correlation_matrix       │
                     │ Ollama Local │     │                              │
                     │ qwen2.5:0.5b │     │  SQL Gen + Synthesis (LLM)  │
                     └──────────────┘     └──────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Docker** installed and running
2. **Ollama** running locally with the model pulled:

```bash
ollama pull qwen2.5:0.5b
ollama serve
```

### Launch

```bash
# Clone or copy the project
cd analytics-app

# Build and run
docker compose up --build

# Open browser
open http://localhost:5000
```

That's it. Upload CSVs and start chatting.

### Without Docker

```bash
pip install -r requirements.txt

# Set Ollama URL for local use
export OLLAMA_URL=http://localhost:11434

python app.py
```

## Usage

### Upload Data
- Drag & drop CSV files onto the sidebar
- Any size, any number of files
- Auto-converted to SQLite tables

### Ask Questions
The agent auto-detects your intent and picks the right tools:

| You Ask | Agent Does |
|---|---|
| "Show revenue by region" | `create_chart` → bar chart |
| "Compare salary across departments" | `compare_groups` → box plot + stats |
| "Find anomalies in revenue" | `detect_anomalies` → outlier detection |
| "Revenue trend over time" | `trend_analysis` → time series |
| "Correlations in employee data" | `correlation_matrix` → heatmap |
| "Profile the sales table" | `profile_table` → full statistical profile |
| "Top 5 products by revenue" | `run_sql` → query + table |
| "Pie chart of customer segments" | `create_chart` → pie chart |

### Chart Types
Bar, line, pie, scatter, histogram, heatmap — auto-detected from your words.

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `MODEL` | `qwen2.5:0.5b` | Ollama model name |
| `MAX_UPLOAD_MB` | `500` | Max upload size in MB |
| `PORT` | `5000` | Server port |

### Using a Bigger Model

For better SQL generation and insights:

```bash
ollama pull qwen2.5:3b
# Then set MODEL=qwen2.5:3b in docker-compose.yml
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/api/upload` | POST | Upload CSVs (multipart) |
| `/api/chat` | POST | Send message, get response |
| `/api/tables` | GET | List loaded tables |
| `/api/clear` | POST | Clear all data |
| `/api/health` | GET | Health check |

## Project Structure

```
analytics-app/
├── app.py              # Flask server + 7 tools + orchestrator
├── templates/
│   └── index.html      # Full frontend (HTML + CSS + JS)
├── static/
│   └── charts/         # Generated chart images
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
