from transformers import pipeline
from app.env import EmailEnv
from app.models import Action

# Load model
generator = pipeline("text-generation", model="distilgpt2")

env = EmailEnv()

total_score = 0

# ===============================
# LOOP THROUGH TASKS
# ===============================
obs = env.reset()
task_id = 1
step_count = 0

print(f"[START] task=email_task", flush=True)

while True:
    step_count += 1

    # Generate response
    result = generator(obs.email_text, max_new_tokens=30)
    reply = result[0]["generated_text"]

    action = Action(response=reply)

    obs, reward, done, _ = env.step(action)

    print(f"[STEP] step={step_count} reward={reward}", flush=True)

    total_score += reward

    if done:
        break

print(f"[END] task=email_task score={total_score} steps={step_count}", flush=True)
