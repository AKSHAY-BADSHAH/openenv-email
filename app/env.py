from app.models import Observation, Action
from app.tasks import task_easy, task_medium, task_hard
from app.grader import grade_easy, grade_medium, grade_hard

class EmailEnv:

    def __init__(self):
        self.tasks = [
            ("easy", task_easy.get_task, grade_easy),
            ("medium", task_medium.get_task, grade_medium),
            ("hard", task_hard.get_task, grade_hard)
        ]
        self.current_task_index = 0
        self.done = False

    def reset(self):
        self.current_task_index = 0
        self.done = False

        task_name, task_func, _ = self.tasks[self.current_task_index]
        self.current_input = task_func()

        return Observation(email_text=self.current_input)

    def step(self, action: Action):
        task_name, task_func, grader = self.tasks[self.current_task_index]

        reward = grader(action.response)

        self.current_task_index += 1

        if self.current_task_index >= len(self.tasks):
            self.done = True
            return Observation(email_text="All tasks completed"), reward, True, {}

        next_task = self.tasks[self.current_task_index][1]()
        return Observation(email_text=next_task), reward, False, {}

    def state(self):
        return {
            "task_index": self.current_task_index,
            "done": self.done
        }