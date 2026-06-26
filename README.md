# SUST Hackathon Preliminary - Setup & Run Guide

This guide walks you through cloning the repository, setting up the environment, running the FastAPI server, and testing all endpoints.

---

# Quick Start (Copy & Paste)

### Terminal 1 — Setup & Start

```bash
git clone https://github.com/tousiftanzeeeem/sust-hackathon-preli.git

cd sust-hackathon-preli

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

echo "OPENROUTER_API_KEY=your_key_here" > .env

uvicorn main:app --port 8002 --reload
```

### Terminal 2 — Test Endpoints

```bash
cd sust-hackathon-preli

source .venv/bin/activate

curl -X GET http://localhost:8002/health

curl -X POST http://localhost:8002/analyze-ticket \
-H "Content-Type: application/json" \
-d '{"ticket_id":"TKT-001","complaint":"Wrong transfer","language":"en","transaction_history":[]}'
```

---


### Detailed Step-by-Step Instructions

---

# Step 1: Clone the Repository

```bash
git clone https://github.com/tousiftanzeeeem/sust-hackathon-preli.git
cd sust-hackathon-preli
```

---

# Step 2: Create a Python Virtual Environment

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Expected Output**

Your terminal prompt should now begin with:

```text
(.venv)
```

---

# Step 3: Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- **fastapi** – Web framework
- **uvicorn** – ASGI server
- **pydantic** – Data validation
- **langchain** & **langchain_openrouter** – LLM integration
- **python-dotenv** – Environment variable management

---

# Step 4: Configure Environment Variables

Create a `.env` file in the project root.

```bash
cat > .env << 'EOF'
OPENROUTER_API_KEY=your_api_key_here
EOF
```

> **Note:** If you don't have an OpenRouter API key, ask your team for one or verify whether the project supports another LLM provider.

---

# Step 5: Verify the Project Structure

```bash
ls -la
```

You should see something similar to:

```text
main.py
README.md
requirements.txt
agent/
middlewares/
tests/
.env
.venv/
```

---

# Step 6: Start the FastAPI Server

```bash
uvicorn main:app --port 8002 --reload
```

Expected output:

```text
INFO:     Uvicorn running on http://127.0.0.1:8002
INFO:     Application startup complete
```

Keep this terminal open.

The `--reload` flag automatically restarts the server whenever you modify the source code.

---

# Step 7: Test the `/health` Endpoint

Open a **new terminal**, activate the virtual environment again, then run:

```bash
cd sust-hackathon-preli
source .venv/bin/activate
```

Test the endpoint:

```bash
curl -X GET http://localhost:8002/health
```

Expected response:

```json
{
  "status": "ok"
}
```

---

# Step 8: Test the `/analyze-ticket` Endpoint

Create a sample request file.

```bash
cat > test_request.json << 'EOF'
{
  "ticket_id": "TKT-001",
  "complaint": "I sent 5000 taka to a wrong number around 2pm today. The number was supposed to be 01712345678 but I think I typed it wrong. The person isn't responding to my call. Please help me get my money back.",
  "language": "en",
  "channel": "in_app_chat",
  "user_type": "customer",
  "campaign_context": "boishakh_bonanza_day_1",
  "transaction_history": [
    {
      "transaction_id": "TXN-9101",
      "timestamp": "2026-04-14T14:08:22Z",
      "type": "transfer",
      "amount": 5000,
      "counterparty": "+8801719876543",
      "status": "completed"
    }
  ]
}
EOF
```

Send the request:

```bash
curl -X POST http://localhost:8002/analyze-ticket \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

Expected response:

```json
{
  "ticket_id": "TKT-001",
  "relevant_transaction_id": "TXN-9101",
  "evidence_verdict": "consistent",
  "case_type": "wrong_transfer",
  "severity": "high",
  "department": "dispute_resolution",
  "agent_summary": "...",
  "recommended_next_action": "...",
  "customer_reply": "...",
  "human_review_required": true,
  "confidence": 0.95,
  "reason_codes": [
    "..."
  ]
}
```

---

# Step 9: Test Using Python (Alternative to cURL)

```python
import httpx
import json

# Health Check
response = httpx.get("http://localhost:8002/health")
print("Health Check:")
print(response.status_code, response.json())
print()

# Analyze Ticket
payload = {
    "ticket_id": "TKT-TEST-001",
    "complaint": "I sent 3000 taka to the wrong person by mistake.",
    "language": "en",
    "transaction_history": [
        {
            "transaction_id": "TXN-001",
            "timestamp": "2026-04-14T10:00:00Z",
            "type": "transfer",
            "amount": 3000,
            "counterparty": "+8801712345678",
            "status": "completed"
        }
    ]
}

response = httpx.post(
    "http://localhost:8002/analyze-ticket",
    json=payload
)

print("Analyze Ticket:")
print(response.status_code)
print(json.dumps(response.json(), indent=2))
```

---

# Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'fastapi'` | Run `pip install -r requirements.txt` again. |
| `Connection refused on localhost:8002` | Ensure the Uvicorn server is still running. |
| `OPENROUTER_API_KEY not set` | Create a `.env` file containing your API key. |
| `422 Unprocessable Entity` | Verify that your request JSON matches the expected schema. |
| `500 Internal Server Error` | Check the Uvicorn logs for the full stack trace. It is often caused by an LLM API configuration issue. |



## 🎉 Done!

Your FastAPI application should now be running successfully, and both the `/health` and `/analyze-ticket` endpoints are ready for testing.