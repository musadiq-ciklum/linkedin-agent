# RAG Chatbot — Assignment 04

This project is part of a step-by-step implementation of a **Retrieval-Augmented Generation (RAG)** system.  
The first completed module is **PDF Processing**, which extracts structured and cleaned text from lecture slides so it can later be used for chunking, embeddings, vector search, and chatbot responses.

---

## 📦 Installation & Setup

### 1. Clone the Repository

**SSH**
```bash
git clone git@github.com:musadiq-ciklum/linkedin-agent.git
```

**HTTPS**
```bash
git clone https://github.com/musadiq-ciklum/linkedin-agent.git
```

**Navigate into the folder:**
```bash
cd linkedin-agent
```
**Create & Activate a Virtual Environment**
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Windows
```bash
python3 -m venv venv
source venv/bin/activate
```


**Install Requirements**
```bash
pip install -r requirements.txt
```

**Run The Tests**
```bash
pytest tests/
```