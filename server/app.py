from fastapi import FastAPI
from app.env import EmailEnv
from app.models import Action
import uvicorn

app = FastAPI()
env = EmailEnv()

@app.get("/")
def home():
    return {"status": "ok"}

# ✅ RESET (ONLY observation)
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": {
            "email_text": obs.email_text
        }
    }

# ✅ STEP (full format)
@app.post("/step")
def step(action: Action):
    obs, reward, done, _ = env.step(action)

    return {
        "observation": {
            "email_text": obs.email_text
        },
        "reward": float(reward),
        "done": bool(done),
        "info": {}
    }

@app.get("/state")
def state():
    return {
        "current_index": env.current_index,
        "total_tasks": len(env.tasks)
    }

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()