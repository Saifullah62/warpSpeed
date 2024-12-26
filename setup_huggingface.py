from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

def setup_huggingface_repo():
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
    
    # Initialize Hugging Face API
    api = HfApi()
    
    try:
        # Create the dataset repository
        repo_id = "GotThatData/warp-speed"
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            token=token
        )
        print(f"Created repository: {repo_id}")
        
        # Upload README
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print("Uploaded README.md")
        
        # Upload dataset card
        api.upload_file(
            path_or_fileobj="dataset-card.md",
            path_in_repo="dataset-card.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print("Uploaded dataset-card.md")
        
        # Create .gitattributes for large file handling
        with open(".gitattributes", "w") as f:
            f.write("*.pdf filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.fits filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.nc filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.grib filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.vtk filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.h5 filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.mat filter=lfs diff=lfs merge=lfs -text\n")
        
        # Upload .gitattributes
        api.upload_file(
            path_or_fileobj=".gitattributes",
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print("Uploaded .gitattributes")
        
        print("\nHugging Face repository setup complete!")
        print(f"Repository URL: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"Error setting up repository: {str(e)}")

if __name__ == "__main__":
    setup_huggingface_repo()
