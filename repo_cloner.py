import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from git import Repo, exc

# Load environment variables
load_dotenv()

def clone_repo(repo_url, branch, target_dir="cloned_repo"):
    """Clone a repository and specific branch"""
    try:
        repo = Repo.clone_from(
            repo_url,
            target_dir,
            branch=branch
        )
        print(f"Cloned {repo_url} ({branch}) to {target_dir}")
        return target_dir
    except exc.GitCommandError as e:
        print(f"Error cloning repository: {e}")
        return None
    
