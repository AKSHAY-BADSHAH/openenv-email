from transformers import pipeline, GenerationConfig
from app.env import EmailEnv
from app.models import Action
from app.logger import log_task

# ===============================
# Load model
# ===============================
generator = pipeline("text-generation", model="distilgpt2")

# Proper generation config (NO warnings)
gen_config = GenerationConfig(
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)

# Fix padding warning
generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

# ===============================
# Initialize Environment
# ===============================
env = EmailEnv()

print("🚀 Starting Email AI Environment...\n")

total_reward_all = 0

# ===============================
# Run multiple episodes
# ===============================
for episode in range(2):

    print(f"\n================ EPISODE {episode + 1} ================\n")

    obs = env.reset()
    total_reward = 0

    while True:
        print("📩 TASK:", obs.email_text)

        # ===============================
        # Smart Prompt Engineering
        # ===============================
        if "Classify" in obs.email_text:
            prompt = f"""
You are an email classifier.

Email:
{obs.email_text}

Answer ONLY one word:
spam or important.
"""

        elif "Extract" in obs.email_text:
            prompt = f"""
You are an email assistant.

Email:
{obs.email_text}

Extract only:
- Name
- Date
- Time

Give short answer like:
John, 15 March, 5 PM
"""

        else:
            prompt = f"""
You are a professional assistant.

Email:
{obs.email_text}

Write a short, polite and professional reply.
"""

        # ===============================
        # Generate Response (NO warnings)
        # ===============================
        result = generator(
            prompt,
            generation_config=gen_config
        )

        # Clean output (remove prompt part)
        reply = result[0]["generated_text"].replace(prompt, "").strip()

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
# Final Result
# ===============================
print("\n================ FINAL RESULT ================")
print("🔥 Total Score:", total_reward_all)
