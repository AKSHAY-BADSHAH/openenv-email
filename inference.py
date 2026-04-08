from transformers import pipeline
from app.env import EmailEnv
from app.models import Action
from app.logger import log_task

# ===============================
# Load Better Model (IMPORTANT)
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

# ===============================
# Run Episodes
# ===============================
for episode in range(2):

    print(f"\n================ EPISODE {episode + 1} ================\n")

    obs = env.reset()
    total_reward = 0

    while True:
        print("📩 TASK:", obs.email_text)

        # ===============================
        # SMART PROMPTS (OPTIMIZED)
        # ===============================
        if "Classify" in obs.email_text:
            prompt = f"""
Classify the following email as spam or important.

{obs.email_text}

Answer only one word: spam or important.
"""

        elif "Extract" in obs.email_text:
            prompt = f"""
Extract name, date, and time from this email.

{obs.email_text}

Answer format:
Name, Date, Time
"""

        elif "Write" in obs.email_text:
            prompt = f"""
Write a short, polite and professional reply to this email.

{obs.email_text}
"""

        else:
            prompt = obs.email_text

        # ===============================
        # GENERATE (CLEAN)
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

        # ===============================
        # Logging
        # ===============================
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

# ===============================
# Final Output
# ===============================
print("\n================ FINAL RESULT ================")
print("🔥 Total Score:", total_reward_all)
