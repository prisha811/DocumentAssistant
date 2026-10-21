FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid downloading NVIDIA CUDA packages.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install the remaining application dependencies.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /app/app/data/raw_pdfs /app/app/data/faiss_index

EXPOSE 8000

CMD ["uvicorn", "api:app", "--app-dir", "/app/app", "--host", "0.0.0.0", "--port", "8000"]