FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ src/
COPY training/ training/
COPY configs/ configs/

ENV PYTHONPATH=/workspace

ENTRYPOINT ["python", "-m", "training.train", "--config"]
CMD ["configs/default.yaml"]
