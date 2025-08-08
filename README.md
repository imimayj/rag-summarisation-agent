# rag-summarisation-agent
POC RAG agent for summarisation of documents

## Getting started

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd rag-summarisation-agent

# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate  

# Install dependencies from lock file
uv sync
```

## Running the Pipeline
```
python -m src.main
```

This should run the entire pipeline and generate the report. Please let me know if you have any questions.
