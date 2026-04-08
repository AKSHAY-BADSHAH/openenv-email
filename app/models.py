from pydantic import BaseModel

# What agent sees
class Observation(BaseModel):
    email_text: str

# What agent does
class Action(BaseModel):
    response: str

# Score
class Reward(BaseModel):
    score: float