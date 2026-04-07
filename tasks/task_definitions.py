# tasks/task_definitions.py

# Minimal stub so HF build works
# Minimal stubs to fix import error

class GroundTruthIssue:
    def __init__(self, clause_id=None, category=None, severity=None):
        self.clause_id = clause_id
        self.category = category
        self.severity = severity


class TaskDefinition:
    def __init__(self, task_id=None, description=None):
        self.task_id = task_id
        self.description = description