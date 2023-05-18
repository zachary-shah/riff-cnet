## run this script after training to upload model and logged images to HuggingFace
from huggingface_hub import login, HfApi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    nargs="?",
    default="",
    help="path to file to upload"
)
parser.add_argument(
    "--filename_in_repo",
    type=str,
    nargs="?",
    default="",
    help="path of file in HF repo to write to"
)
parser.add_argument(
    "--repo_id",
    type=str,
    nargs="?",
    default="zachary-shah/riffusion-cnet-v3",
    help="repo to upload to"
)
args = parser.parse_args()

# log into hugginface
login()
api = HfApi()

try:
    # try uploading all checkpoints first
    print(f"Uploading file {args.filename} to {args.filename_in_repo}...")
    api.upload_file(
        path_or_fileobj=args.filename,
        path_in_repo=args.filename_in_repo,
        repo_id=args.repo_id,
    )
    print("File uploaded!")

except:
    # just get most recently created checkpoint in case of some storage issue
    print("Upload failed.")
