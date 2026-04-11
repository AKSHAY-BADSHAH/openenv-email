<<<<<<< HEAD
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
=======
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
>>>>>>> a9bb911 (final phase 2 fix)
    run()