from fastapi import FastAPI
from app.env import EmailEnv
from app.models import Action

app = FastAPI()
env = EmailEnv()

# ROOT
@app.get("/")
def home():
    return {"message": "OpenEnv Email Environment Running"}

# RESET
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.model_dump()   # ✅ FIXED
    }

# STEP
@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),  # ✅ FIXED
        "reward": float(reward),          # ✅ ensure float
        "done": bool(done),               # ✅ ensure bool
        "info": info or {}                # ✅ never None
    }

# STATE
@app.get("/state")
def state():
    return env.state()
