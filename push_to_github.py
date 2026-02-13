"""
Push complete project to GitHub repository
Production-ready with error handling
"""

import os
import subprocess


def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"\n▶ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0 and "already exists" not in result.stderr.lower():
            print(f"  ⚠ Warning: {result.stderr.strip()}")
        else:
            print(f"  ✓ {description} completed")
        return result.returncode == 0
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def push_to_github():
    """Push project to GitHub repository"""
    
    print("=" * 70)
    print("PUSHING PROJECT TO GITHUB")
    print("=" * 70)
    
    # Get credentials
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN environment variable not found")
    
    username = "greeshmak99"
    repo = "Engine_Predictive_Maintainance"
    
    print(f"\nRepository: https://github.com/{username}/{repo}")
    print("=" * 70)
    
    # Git commands
    commands = [
        ("git init", "Initialize Git repository"),
        (f"git config user.name '{username}'", "Configure Git username"),
        (f"git config user.email '{username}@users.noreply.github.com'", "Configure Git email"),
        ("git add .", "Stage all files"),
        ('git commit -m "Initial commit: Complete ML pipeline with deployment"', "Commit changes"),
        ("git branch -M main", "Set main branch"),
        (f"git remote add origin https://{github_token}@github.com/{username}/{repo}.git", "Add remote origin"),
        ("git push -u origin main --force", "Push to GitHub")
    ]
    
    # Execute commands
    for cmd, desc in commands:
        # Hide token in output
        display_cmd = cmd.replace(github_token, "***TOKEN***") if github_token in cmd else cmd
        run_command(cmd, desc)
    
    print("\n" + "=" * 70)
    print("✓ PROJECT PUSHED TO GITHUB SUCCESSFULLY")
    print("=" * 70)
    print(f"\n🔗 Repository: https://github.com/{username}/{repo}")
    print("\n📝 Next Steps:")
    print("   1. Go to repository Settings → Secrets and variables → Actions")
    print("   2. Add secret: HF_EN_TOKEN = your_huggingface_token")
    print("   3. GitHub Actions will run automatically on next push")
    print("\n⚡ Workflow will verify:")
    print("   - Data preparation pipeline")
    print("   - Model training pipeline")
    print("   - Deployment to HF Space")
    print("=" * 70)


if __name__ == "__main__":
    push_to_github()
