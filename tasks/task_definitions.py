
class GroundTruthIssue:
    def __init__(self, clause_id=None, category=None, severity=None):
        self.clause_id = clause_id
        self.category = category
        self.severity = severity


class TaskDefinition:
    def __init__(self, task_id=None, description=None):
        self.task_id = task_id
        self.description = description


# FIX: define ALL_TASKS 
ALL_TASKS = {
    "task_easy_freelance": TaskDefinition(
        task_id="task_easy_freelance",
        description="Example task"
    )
}