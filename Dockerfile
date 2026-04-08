FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
RUN find /app -name "*.pyc" -delete 2>/dev/null || true

RUN touch env/__init__.py tasks/__init__.py graders/__init__.py

RUN python -c "import sys; sys.path.insert(0,'/app'); from tasks.task_definitions import ALL_TASKS; tasks=list(ALL_TASKS.values()); assert len(tasks)==3; [print('OK:',t.task_id,t.difficulty) for t in tasks]; print('All 3 tasks verified!')"

ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]