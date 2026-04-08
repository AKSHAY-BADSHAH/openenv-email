from transformers import pipeline
from app.env import EmailEnv
from app.models import Action

generator = pipeline("text-generation", model="distilgpt2")

env = EmailEnv()

obs = env.reset()
total = 0

while True:
    print("TASK:", obs.email_text)

    result = generator(obs.email_text, max_new_tokens=30)
    reply = result[0]["generated_text"]

    print("AI:", reply)

    action = Action(response=reply)

    obs, reward, done, _ = env.step(action)

    print("Reward:", reward)
    total += reward

    if done:
        break

print("Final Score:", total)
