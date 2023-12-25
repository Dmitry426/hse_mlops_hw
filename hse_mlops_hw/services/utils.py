import subprocess
import sys


def run_dvc_pull():
    """Dummy dvc pull command"""
    try:
        subprocess.run(["dvc", "pull"], check=True)
        print("DVC pull completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: DVC pull failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except OSError as e:
        print(f"Error: An unexpected error occurred - {e}")
        sys.exit(1)
