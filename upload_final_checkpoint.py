## run this script after training to upload model and logged images to HuggingFace
from huggingface_hub import login, HfApi
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--control_method",
    type=str,
    nargs="?",
    default="canny",
    help="method of control used for filepath naming"
)
parser.add_argument(
    "--image_log_dir",
    type=str,
    nargs="?",
    default="image_log/",
    help="root directory where all image log data is"
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    nargs="?",
    default="models/riff-cnet-final.ckpt",
    help="path of model to upload"
)
parser.add_argument(
    "--repo_id",
    type=str,
    nargs="?",
    default="zachary-shah/riff-cnet-experiments",
    help="repo to upload to"
)
args = parser.parse_args()

# log into hugginface
login()
api = HfApi()
root_dir = os.getcwd()

# upload image logs
try:
    print("Uploading images...")
    api.upload_folder(
        folder_path=args.image_log_dir,
        path_in_repo=os.path.join("image_logs", args.control_method+"_image_log/"),
        repo_id=args.repo_id,
    )
    print("Images upoaded.")
except:
    print("Image upload failed.")

try:
    # try uploading all checkpoints first
    print("Uploading checkpoint...")
    api.upload_file(
        path_or_fileobj=args.ckpt_path,
        path_in_repo=os.path.join("checkpoints",args.control_method+"-"+args.ckpt_path.split("/")[-1]),
        repo_id=args.repo_id,
    )
    print("Checkpoint uploaded!")

except:
    # just get most recently created checkpoint in case of some storage issue
    print("Upload failed. uploading only the last checkpoint:")