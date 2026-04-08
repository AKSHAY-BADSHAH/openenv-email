from transformers import pipeline
from app.env import EmailEnv
from app.models import Action
from app.logger import log_task

# Load free model
generator = pipeline("text-generation", model="distilgpt2")

env = EmailEnv()

print("🚀 Starting Email AI Environment...\n")

total_reward_all = 0

# 🔁 Multiple runs (STEP 4)
for episode in range(2):

    print(f"\n================ EPISODE {episode + 1} ================\n")

    obs = env.reset()
    total_reward = 0

    while True:
        print("📩 TASK:", obs.email_text)

        result = generator(
            obs.email_text,
            max_length=60,
            num_return_sequences=1
        )

        reply = result[0]['generated_text']

        print("🤖 AI:", reply)

        action = Action(response=reply)

        obs, reward, done, _ = env.step(action)

        print("🏆 Reward:", reward)

        # ✅ Logging
        log_task(obs.email_text if not done else "FINAL", reply, reward)

        total_reward += reward

        if done:
            break

    print(f"\n🎯 Episode Score: {total_reward}")
    total_reward_all += total_reward

print("\n================ FINAL RESULT ================")
print("🔥 Total Score:", total_reward_all)
