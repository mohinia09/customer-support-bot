# Customer Support Email Automation

An intelligent email automation system that handles customer support queries at scale by classifying incoming emails, drafting accurate replies from a knowledge base, and routing complex conversations to human agents.

---

## Problem

Support teams spend significant time triaging and responding to repetitive customer queries. Most of these questions have known answers in existing FAQ and policy documents, yet they still require manual effort to respond to individually. At scale, this creates delays in response time and diverts human attention from complex issues that genuinely need it.

---

## Solution

This system automates the first line of customer support by continuously monitoring an inbox, understanding the nature of each email, and responding accurately using a curated knowledge base. Conversations that are complex, multi-turn, or outside the scope of available documentation are automatically flagged for human review ensuring no customer falls through the cracks.

---

## Key Features

- Continuous inbox monitoring with configurable polling interval
- Automatic classification of incoming emails
- AI-generated replies grounded in your knowledge base documents
- Automatic escalation of multi-turn conversations to human agents
- Escalation when no confident answer is found in the knowledge base
- Gmail label-based tracking, no external database required
- Duplicate processing prevention across runs

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.13 |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| AI Agent & Classifier | LangChain |
| Vector Store | ChromaDB |
| Email | Gmail API |
| Scheduler | APScheduler |

---

## Project Structure

```
â”œâ”€â”€ main.py               # Entry point and orchestration
â”œâ”€â”€ gmail_client.py       # Gmail integration
â”œâ”€â”€ classifier.py         # Email classification
â”œâ”€â”€ agent.py              # AI agent and knowledge base retrieval
â”œâ”€â”€ vector_store.py       # Vector store setup
â”œâ”€â”€ knowledge_base/       # FAQ and policy documents (.txt)
â”œâ”€â”€ .env                  # Configuration (not committed)
â””â”€â”€ requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/customer-support-bot.git
cd customer-support-bot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_CREDENTIALS_FILE=credentials.json
EMAIL_POLL_INTERVAL_MINUTES=2
```

### 3. Gmail API setup

1. Enable Gmail API in Google Cloud Console
2. Create OAuth 2.0 credentials (Desktop App)
3. Download and rename to `credentials.json`
4. Place in project root

### 4. Add knowledge base

Place `.txt` files containing FAQ and policy documents in the `knowledge_base/` folder.

### 5. Run

```bash
python main.py
```

---

## Roadmap

- [ ] Rate limiting and cost controls
- [ ] Human escalation notifications
- [ ] Production deployment with Service Account
- [ ] Logging and monitoring
