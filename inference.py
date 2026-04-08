from transformers import pipeline
from app.env import EmailEnv
from app.models import Action
from app.logger import log_task

# ===============================
# Correct pipeline for FLAN-T5
# ===============================
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# ===============================
# Initialize Environment
# ===============================
env = EmailEnv()

print("🚀 Starting Email AI Environment...\n")

total_reward_all = 0

for episode in range(2):

    print(f"\n================ EPISODE {episode + 1} ================\n")

    obs = env.reset()
    total_reward = 0

    while True:
        print("📩 TASK:", obs.email_text)

        # ===============================
        # SMART PROMPTS
        # ===============================
        if "Classify" in obs.email_text:
            prompt = f"Classify this email as spam or important:\n{obs.email_text}"

        elif "Extract" in obs.email_text:
            prompt = f"Extract name, date, and time from this email:\n{obs.email_text}"

        elif "Write" in obs.email_text:
            prompt = f"Write a short professional reply:\n{obs.email_text}"

        else:
            prompt = obs.email_text

        # ===============================
        # GENERATE (CORRECT WAY)
        # ===============================
        result = generator(
            prompt,
            max_new_tokens=50
        )

        reply = result[0]["generated_text"].strip()

        print("🤖 AI:", reply)

        # ===============================
        # Environment Step
        # ===============================
        action = Action(response=reply)
        obs, reward, done, _ = env.step(action)

        print("🏆 Reward:", reward)

        log_task(
            obs.email_text if not done else "FINAL",
            reply,
            reward
        )

        total_reward += reward

        if done:
            break

    print(f"\n🎯 Episode Score: {total_reward}")
    total_reward_all += total_reward

print("\n================ FINAL RESULT ================")
print("🔥 Total Score:", total_reward_all)
