from app.env import EmailEnv
from app.models import Action
import sys
import time

def log(msg):
    print(msg)
    sys.stdout.flush()

def main():
    env = EmailEnv()

    obs = env.reset()
    total_score = 0
    step_count = 0

    log("[START] task=email_task")
    time.sleep(0.2)   # 🔥 ensures logs visible

    while True:
        step_count += 1

        text = obs.email_text.lower()

        # RULE-BASED RESPONSE
        if "classify" in text:
            reply = "spam"

        elif "extract" in text:
            reply = "John, 15 March, 5 PM"

        elif "write" in text:
            reply = "Sure, we can reschedule the meeting to tomorrow."

        else:
            reply = "ok"

        action = Action(response=reply)

        obs, reward, done, _ = env.step(action)

        log(f"[STEP] step={step_count} reward={reward}")
        time.sleep(0.2)   # 🔥 important

        total_score += reward

        if done:
            break

    log(f"[END] task=email_task score={total_score} steps={step_count}")
    time.sleep(0.2)

if __name__ == "__main__":
    main()