from transformers import pipeline
from app.env import EmailEnv
from app.models import Action
from app.logger import log_task

# Load model
generator = pipeline("text-generation", model="distilgpt2")

# Fix padding warning
generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

env = EmailEnv()

print("🚀 Starting Email AI Environment...\n")

total_reward_all = 0

for episode in range(2):

    print(f"\n================ EPISODE {episode + 1} ================\n")

    obs = env.reset()
    total_reward = 0

    while True:
        print("📩 TASK:", obs.email_text)

        # 🔥 SMART PROMPT (fixes all issues)
        if "Classify" in obs.email_text:
            prompt = f"""
You are an email classifier.

{obs.email_text}

Answer ONLY one word: spam or important.
"""
        elif "Extract" in obs.email_text:
            prompt = f"""
You are an email assistant.

{obs.email_text}

Extract only key details: name, date, time.
Give short answer.
"""
        else:
            prompt = f"""
You are a professional assistant.

{obs.email_text}

Write a short, polite, professional reply.
"""

        # ✅ CLEAN GENERATION (no warnings)
        result = generator(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )

        reply = result[0]["generated_text"]

        print("🤖 AI:", reply)

        action = Action(response=reply)

        obs, reward, done, _ = env.step(action)

        print("🏆 Reward:", reward)

        # Logging
        log_task(obs.email_text if not done else "FINAL", reply, reward)

        total_reward += reward

        if done:
            break

    print(f"\n🎯 Episode Score: {total_reward}")
    total_reward_all += total_reward

print("\n================ FINAL RESULT ================")
print("🔥 Total Score:", total_reward_all)
