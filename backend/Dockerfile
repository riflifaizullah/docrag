FROM python:3.10-slim

# Tesseract (English and Indonesian) and Poppler are only used for scanned PDFs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-ind poppler-utils libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# The CPU index alone lacks torch's own dependencies, so PyPI is added as a second index.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Documents, the index and logs live in /data. Model downloads are cached in /hf.
ENV DOCRAG_DATA_DIR=/data HF_HOME=/hf
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
