FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements_docker.txt .
RUN pip install torch==2.11.0+cu126 torchvision==0.26.0+cu126 \
    --extra-index-url https://download.pytorch.org/whl/cu126

RUN pip install -r requirements_docker.txt

EXPOSE 8000

CMD ["uvicorn", "app_chain:app", "--host", "0.0.0.0", "--port", "8000"]