# 🎓 SaiU Academic Chatbot

A RAG-based (Retrieval-Augmented Generation) conversational chatbot built for **Sai University** students and faculty. Ask it anything about courses, faculty, policies, scholarships, deadlines, and more — and get instant, accurate answers drawn from official university documents.

🔗 **Live Demo**: [https://famous-banoffee-110f75.netlify.app](https://famous-banoffee-110f75.netlify.app)  
📦 **GitHub**: [https://github.com/ishaw015/saiu-chatbot](https://github.com/ishaw015/saiu-chatbot)

---

## ✨ Features

- **Course queries** — Find courses offered by any school (SCDS, SOL, SOB, SOAI, SAS, SOT, Media, Technology)
- **Faculty lookup** — Ask who teaches any course, case-insensitive
- **Academic policies** — Add/drop deadlines, grading policies, attendance rules
- **Scholarship info** — CGPA requirements for each scholarship tier
- **Holiday calendar** — Full list of holidays for 2026
- **School & curriculum structure** — Program details across all 8 schools
- **Dual API key fallback** — Automatically switches Groq API keys under rate limits
- **Intent detection** — Understands natural language queries without keyword matching

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Plain HTML, CSS, JavaScript |
| **Backend** | FastAPI (Python 3.11) |
| **Vector Store** | ChromaDB (2252 chunks) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **LLM** | Groq `llama-3.3-70b-versatile` |
| **Frontend Hosting** | Netlify |
| **Backend Hosting** | Railway |

---

## 📁 Project Structure

```
saiu-chatbot/
├── src/
│   ├── app.py           # Main FastAPI app, endpoints, retrieval logic, LLM calls
│   ├── vector_store.py  # Builds ChromaDB from source documents
│   ├── retriever.py     # ChromaDB retrieval with school tag filtering
│   ├── chunking.py      # Document chunking logic
│   └── data_loader.py   # Document classification, OCR for scanned PDFs
├── data/                # 16 university documents (catalogue, policies, holidays)
├── chroma_db/           # Pre-built vector store (2252 chunks)
├── model_cache/         # Cached HuggingFace embeddings
├── scripts/
│   └── startup.py       # Rebuilds vector store if missing on server
├── saiu-chatbot.html    # Single-file frontend (for local testing)
├── Procfile             # Railway start command
├── railway.toml         # Railway deployment config
├── runtime.txt          # Python version pin
└── requirements.txt     # Python dependencies
```

---

## 💬 Example Queries

| Query | Response |
|-------|----------|
| "Which faculty is teaching Programming in Python?" | Ujjwal, Arjun, David, Vivekanand |
| "What courses are offered in SCDS this semester?" | Full SCDS course list |
| "What is the deadline to add or drop a course?" | 01 February 2026 |
| "What CGPA do I need for 50% scholarship?" | Minimum CGPA of 8 |
| "What are the holidays in 2026?" | Full holiday list |
| "What is the fee structure for BBA?" | Redirects to registrar |

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11
- A [Groq API key](https://console.groq.com) (free)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ishaw015/saiu-chatbot.git
   cd saiu-chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv311
   # Windows
   venv311\Scripts\activate
   # Mac/Linux
   source venv311/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_API_KEY_2=your_second_groq_api_key_here
   ```

5. **Build the vector store** *(skip if `chroma_db/` already exists)*
   ```bash
   python -m src.vector_store
   ```

6. **Start the backend**
   ```bash
   python -m src.app
   ```
   Backend runs at `http://localhost:8000`

7. **Open the frontend**

   Open `saiu-chatbot.html` in your browser. Make sure the API URL in the file points to `http://localhost:8000/chat`.

---

## ☁️ Deployment Guide

### Backend — Railway

1. Push your code to a GitHub repository
2. Go to [railway.app](https://railway.app) and sign in with GitHub
3. Click **New Project** → **Deploy from GitHub repo** → select your repo
4. Go to **Variables** tab and add:
   - `GROQ_API_KEY`
   - `GROQ_API_KEY_2`
   - `PORT` = `8000`
5. Go to **Networking** tab → click **Generate Domain**
6. Railway auto-deploys on every push to `main`

> **Note:** The `chroma_db/` and `model_cache/` folders are committed to the repo (~120MB total) so Railway uses them directly without rebuilding.

### Frontend — Netlify

1. Open `saiu-chatbot.html` and update the API URL:
   ```javascript
   const API_URL = 'https://your-railway-url.up.railway.app/chat'
   ```
2. Go to [netlify.com](https://netlify.com) and log in
3. Drag and drop `saiu-chatbot.html` onto the Netlify dashboard
4. Your site is live instantly at a `*.netlify.app` URL

### Lock CORS (recommended)

Once deployed, update `src/app.py` to restrict CORS to your Netlify domain:
```python
allow_origins=["https://your-site.netlify.app"]
```

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Primary Groq API key |
| `GROQ_API_KEY_2` | Fallback Groq API key (for rate limit handling) |
| `PORT` | Server port (set to `8000`) |

---

## 📚 Data Sources

The chatbot is trained on 16 official Sai University documents including:
- Course catalogue with full course listings per school
- Academic policies (grading, attendance, add/drop deadlines)
- Scholarship eligibility criteria
- Holiday calendar 2026
- Program structure and curriculum documents

---

## 🏫 Schools Supported

SCDS · SOL · SOB · SOAI · SAS · SOT · Media · Technology

---

## 👩‍💻 Built By

Isha Wagh — Sai University
