from transformers import pipeline
from app.env import EmailEnv
from app.models import Action
from app.logger import log_task

# ===============================
# Load Model (FIXED VERSION)
# ===============================
generator = pipeline(
    "text-generation",   # ✅ supported task
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
        # SMART PROMPTS
        # ===============================
        if "Classify" in obs.email_text:
            prompt = f"Classify email as spam or important:\n{obs.email_text}\nAnswer:"

        elif "Extract" in obs.email_text:
            prompt = f"Extract name, date, time:\n{obs.email_text}\nAnswer:"

        elif "Write" in obs.email_text:
            prompt = f"Write a short professional reply:\n{obs.email_text}\nReply:"

        else:
            prompt = obs.email_text

        # ===============================
        # GENERATE OUTPUT
        # ===============================
        result = generator(
            prompt,
            max_new_tokens=50,
            do_sample=False
        )

        raw_output = result[0]["generated_text"]

        # Clean output
        reply = raw_output.replace(prompt, "").strip().split("\n")[0]

        # ===============================
        # FALLBACK FIX (IMPORTANT)
        # ===============================
        if "Classify" in obs.email_text:
            if "spam" in reply.lower():
                reply = "spam"
            elif "important" in reply.lower():
                reply = "important"
            else:
                reply = "spam"

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
