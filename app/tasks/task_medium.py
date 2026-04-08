from app.models import Observation

def get_task():
    return Observation(
        email_text="Extract important details:\nMeeting with John on 15 March at 5 PM."
    )
