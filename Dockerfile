# CPU image for the continuous-batching server. For trained weights set INFER_PRETRAINED=1
# (downloads gpt2 from Hugging Face on start; mount ~/.cache/huggingface to keep it).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000

ARG TORCH_VERSION=2.14.1
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
COPY requirements/lock-serve-py312.txt requirements/
RUN pip install --no-cache-dir -r requirements/lock-serve-py312.txt
COPY pyproject.toml README.md ./
COPY infer ./infer
RUN pip install --no-cache-dir --no-deps . && useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "infer.serving.server"]
