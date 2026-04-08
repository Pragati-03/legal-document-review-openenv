FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Delete ALL pycache so nothing stale is used
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
RUN find /app -name "*.pyc" -delete 2>/dev/null || true

# Ensure __init__.py files exist
RUN touch env/__init__.py tasks/__init__.py graders/__init__.py

# Force Python to recompile everything fresh
RUN python -c "
import compileall
compileall.compile_dir('/app/env', quiet=1)
compileall.compile_dir('/app/tasks', quiet=1)
compileall.compile_dir('/app/graders', quiet=1)
print('Compilation done')
"

# Verify tasks load correctly at build time - will fail build if broken
RUN python -c "
import sys
sys.path.insert(0, '/app')
from tasks.task_definitions import ALL_TASKS
assert len(ALL_TASKS) == 3, f'Expected 3 tasks, got {len(ALL_TASKS)}'
for tid, t in ALL_TASKS.items():
    assert hasattr(t, 'difficulty'), f'Task {tid} missing difficulty'
    assert hasattr(t, 'document_title'), f'Task {tid} missing document_title'
    print(f'OK: {tid} - {t.difficulty} - {t.document_title}')
print('All tasks verified!')
"

ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]