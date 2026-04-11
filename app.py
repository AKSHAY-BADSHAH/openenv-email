import subprocess
import sys

def run():
    process = subprocess.Popen(
        [sys.executable, "inference.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    process.wait()

if __name__ == "__main__":
    run()