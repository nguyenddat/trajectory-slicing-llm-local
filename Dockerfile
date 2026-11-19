FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/nltk_data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader -d "$NLTK_DATA" stopwords

COPY src/ ./src/
COPY data/gold/ ./data/gold/

# Fail the build if Git LFS supplied a pointer instead of the required dataset.
RUN python -c "from pathlib import Path; p = Path('data/gold/trace_elephant/trajectories.jsonl'); f = p.open('rb'); assert not f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'), 'Run git lfs pull before docker build'"

ENTRYPOINT ["python", "src/run_local.py"]
