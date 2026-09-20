FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY toolcall_bench ./toolcall_bench
RUN python -m pip install --no-cache-dir '.[service]' \
    && useradd --create-home --uid 10001 appuser

USER appuser
EXPOSE 8000
CMD ["uvicorn", "toolcall_bench.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
