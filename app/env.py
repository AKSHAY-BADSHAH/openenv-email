from app.tasks.task_easy import get_task as easy
from app.tasks.task_medium import get_task as medium
from app.tasks.task_hard import get_task as hard
from app.grader import grade

class EmailEnv:

    def __init__(self):
        self.tasks = [easy(), medium(), hard()]
        self.current_index = 0

    # ✅ RESET
    def reset(self):
        self.current_index = 0
        return self.tasks[self.current_index]

    # ✅ STEP
    def step(self, action):
        task = self.tasks[self.current_index]

        reward = grade(task.email_text, action.response)

        self.current_index += 1
        done = self.current_index >= len(self.tasks)

        if not done:
            next_obs = self.tasks[self.current_index]
        else:
            next_obs = task  # last observation

        return next_obs, float(reward), bool(done), {}

    # ✅ STATE (IMPORTANT FOR VALIDATION)
    def state(self):
        return {
            "current_index": self.current_index,
            "total_tasks": len(self.tasks)
        }
