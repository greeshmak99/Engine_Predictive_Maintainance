"""
Deploy Streamlit application to Hugging Face Space
Production-ready deployment script
"""

from huggingface_hub import HfApi, login
import os


def deploy_to_hf_space():
    """Deploy to Hugging Face Space"""
    
    print("=" * 70)
    print("DEPLOYING TO HUGGING FACE SPACE")
    print("=" * 70)
    
    # Get token
    hf_token = os.getenv("HF_EN_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_EN_TOKEN environment variable not found")
    
    # Authenticate
    login(token=hf_token)
    print("✓ Authenticated with Hugging Face\n")
    
    # Initialize API
    api = HfApi(token=hf_token)
    space_id = "Quantum9999/Engine-Predictive-Maintenance"
    
    # Create space
    print(f"Creating/verifying Space: {space_id}")
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="streamlit",
            exist_ok=True,
            private=False
        )
        print(f"✓ Space verified/created: {space_id}\n")
    except Exception as e:
        print(f"Note: {e}\n")
    
    # Upload deployment files
    print("Uploading deployment files...")
    print("-" * 70)
    
    try:
        api.upload_folder(
            folder_path="deployment",
            repo_id=space_id,
            repo_type="space",
            path_in_repo="",
            ignore_patterns=[".gitignore", "__pycache__", "*.pyc"]
        )
        print("✓ All deployment files uploaded successfully")
    except Exception as e:
        print(f"✗ Upload error: {e}")
        return
    
    print("\n" + "=" * 70)
    print("DEPLOYMENT SUCCESSFUL!")
    print("=" * 70)
    print(f"\n🔗 Your application URL:")
    print(f"   https://huggingface.co/spaces/{space_id}")
    print("\n📝 Next Steps:")
    print("   1. Go to Space settings")
    print("   2. Add 'HF_TOKEN' secret with your Hugging Face token")
    print("   3. Space will automatically rebuild and start")
    print("\n⏳ Build time: ~2-3 minutes")
    print("=" * 70)


if __name__ == "__main__":
    deploy_to_hf_space()
