from fastapi import FastAPI
from app.env import EmailEnv
from app.models import Action

app = FastAPI()
env = EmailEnv()

# ✅ ROOT (required)
@app.get("/")
def home():
    return {"status": "ok"}

# ✅ RESET
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.model_dump()
    }

# ✅ STEP
@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": {}
    }

# ✅ STATE
@app.get("/state")
def state():
    return env.state()
