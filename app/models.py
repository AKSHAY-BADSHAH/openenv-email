from pydantic import BaseModel

class Observation(BaseModel):
    email_text: str

class Action(BaseModel):
<<<<<<< HEAD
    response: str
=======
    response: str
>>>>>>> a9bb911 (final phase 2 fix)
