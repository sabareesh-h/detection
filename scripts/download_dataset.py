import argparse
import os
from roboflow import Roboflow

def download_dataset(api_key, workspace, project, version=1):
    """
    Downloads the dataset from Roboflow.
    """
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)
        dataset = project.version(version).download("yolov11") # Using yolov11 format as per requirements
        
        # Move dataset to data dir if needed or it will download to current dir
        # Roboflow usually downloads to a folder named after the project
        print(f"Dataset downloaded successfully to: {dataset.location}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Roboflow")
    parser.add_argument("--api-key", type=str, help="Roboflow API Key", required=False)
    parser.add_argument("--workspace", type=str, default="cracks-detection-yrupe", help="Roboflow Workspace")
    parser.add_argument("--project", type=str, default="road-damage-detection-zfqfj", help="Roboflow Project")
    parser.add_argument("--version", type=int, default=1, help="Dataset Version")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    
    if not api_key:
        print("Error: Roboflow API Key is required. Provide it via --api-key or ROBOFLOW_API_KEY environment variable.")
        # Attempt to ask for input interactively if no key provided
        try:
            api_key = input("Please enter your Roboflow API Key: ").strip()
            if not api_key:
                print("No API key provided. Exiting.")
                exit(1)
        except EOFError:
            print("Non-interactive mode detected. Exiting.")
            exit(1)
            
    download_dataset(api_key, args.workspace, args.project, args.version)
