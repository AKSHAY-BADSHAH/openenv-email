from pydantic import BaseModel

class Observation(BaseModel):
    email_text: str

class Action(BaseModel):
    response: str
