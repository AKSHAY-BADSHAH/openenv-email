from app.models import Observation

def get_task():
    return Observation(
        email_text="Write a professional reply:\nCan we reschedule meeting to tomorrow?"
    )
