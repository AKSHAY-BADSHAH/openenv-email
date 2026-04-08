def log_task(task, response, reward):
    with open("logs.txt", "a") as f:
        f.write(f"TASK: {task}\n")
        f.write(f"RESPONSE: {response}\n")
        f.write(f"REWARD: {reward}\n")
        f.write("=" * 40 + "\n")