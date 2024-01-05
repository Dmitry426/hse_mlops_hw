__all__ = "run_dvc_pull"

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def run_dvc_pull():
    """Dummy dvc pull command"""
    try:
        subprocess.run(["dvc", "pull"], check=True)
        logger.info("DVC pull completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: DVC pull failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except OSError as e:
        logger.error(f"Error: An unexpected error occurred - {e}")
        sys.exit(1)
