FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

# LightGBM's wheel needs OpenMP at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Editable install keeps PROJECT_ROOT at /app (src/sbahn/config.py is two levels down).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY reports ./reports
RUN pip install --no-cache-dir -e .

COPY pipeline.sh ./pipeline.sh
RUN sed -i 's/\r$//' pipeline.sh && chmod +x pipeline.sh

ENV GIT_PYTHON_REFRESH=quiet

CMD ["./pipeline.sh"]
