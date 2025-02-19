from github import Github
import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

g = Github(TOKEN)

# Get the repository
repo_user = input("Enter the owner and repository name in the following format: owner/repo_name\n")
repo = g.get_repo(repo_user)

# Get all branches
branches = list(repo.get_branches())

# Collect commit stats per collaborator per branch
data = []
for branch in branches:
    commits = repo.get_commits(sha=branch.name)
    for commit in commits:
        author = commit.author.login if commit.author else "Unknown"
        data.append({
            "Branch": branch.name,
            "Author": author,
            "Commit SHA": commit.sha,
            "Date": commit.commit.author.date
        })

# Convert to DataFrame
df = pd.DataFrame(data)
stats = df.groupby(['Branch', 'Author']).size().reset_index(name='Commits')
print(stats)

def generate_visualizations(stats, output_path='.'):
    # Create figure directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Plot 1: Total commits by branch
    plt.figure(figsize=(10, 6))
    branch_commits = stats.groupby('Branch')['Commits'].sum()
    sns.barplot(x=branch_commits.index, y=branch_commits.values)
    plt.title('Total Commits by Branch')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'commits_by_branch.png'))
    plt.close()
    
    # Plot 2: Distribution of commits per author
    plt.figure(figsize=(10, 6))
    author_commits = stats.groupby('Author')['Commits'].sum()
    sns.barplot(x=author_commits.index, y=author_commits.values)
    plt.title('Total Commits by Author')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'commits_by_author.png'))
    plt.close()

generate_visualizations(stats, 'figures')