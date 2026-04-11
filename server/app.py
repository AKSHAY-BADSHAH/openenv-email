from fastapi import FastAPI
from app.env import EmailEnv
from app.models import Action
import uvicorn

app = FastAPI()
env = EmailEnv()

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.model_dump()}

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": {}
    }

@app.get("/state")
def state():
    return env.state()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
