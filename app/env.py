from app.tasks.task_easy import get_task as easy
from app.tasks.task_medium import get_task as medium
from app.tasks.task_hard import get_task as hard
from app.grader import grade

class EmailEnv:

    def __init__(self):
        self.tasks = [easy(), medium(), hard()]
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.tasks[self.current_index]

    def step(self, action):
        task = self.tasks[self.current_index]

        reward = grade(task.email_text, action.response)

        self.current_index += 1
        done = self.current_index >= len(self.tasks)

        if not done:
            obs = self.tasks[self.current_index]
        else:
            obs = task

        return obs, float(reward), bool(done), {}

    def state(self):
        return {
            "current_index": self.current_index,
            "total_tasks": len(self.tasks)
        }
