[project]
name = "legal-review-openenv"
version = "1.0.0"
description = "AI agent environment for reviewing legal contracts"
requires-python = ">=3.10"
dependencies = [
    "openenv-core>=0.2.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
    "requests>=2.31.0",
]

[project.scripts]
server = "server:app"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends.legacy:build"

[tool.openenv]
name = "Legal Document Review"
version = "1.0.0"
entry_point = "inference.py"
reset_endpoint = "/reset"
step_endpoint = "/step"
observation_endpoint = "/state"
health_endpoint = "/health"
tasks_endpoint = "/tasks"

[tool.openenv.metadata]
task_type = "document_review"
domain = "legal"
modality = "text"
multi_step = true