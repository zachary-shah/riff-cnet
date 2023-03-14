## run this script after training to upload model and logged images to HuggingFace

from huggingface_hub import login, HfApi
import os, sys
from stat import S_ISREG, ST_CTIME, ST_MODE

# pass in checkpoint version number 
version = int(sys.argv[1])
print(f"version: {version}")
repo_id = sys.argv[2]

# log into hugginface
login()
api = HfApi()
root_dir = os.getcwd()

# upload image logs
print("Uploading images...")
print(os.path.join(root_dir, "image_log/"))
api.upload_folder(
    folder_path=os.path.join(root_dir, "image_log/"),
    repo_id=repo_id,
)
print("Images upoaded.")

try:
    # try uploading all checkpoints first
    print("trying to upload all checkpoints...")
    ckpt_path = os.path.join(root_dir, f"lightning_logs/version_{version}/")
    api.upload_folder(
        folder_path=ckpt_path,
        repo_id=repo_id,
    )
    print("All checkpoints uploaded!")

except:
    try:
        # just get most recently created checkpoint in case of some storage issue
        print("upload failed. uploading only the last checkpoint:")
        ckpt_path = os.path.join(root_dir, f"lightning_logs/version_{version}/checkpoints/")
        latest_file = ""
        latest_path = ""
        latest_time = 0
        for file_name in os.listdir(ckpt_path):
            path = os.path.join(ckpt_path, file_name)
            stat = os.stat(path)
            if S_ISREG(stat[ST_MODE]) and (file_name.endswith(".ckpt") or file_name.endswith(".tar")):
                if stat[ST_CTIME] > latest_time:
                    latest_file = file_name
                    latest_path = path

        print(f"Saving checkpoint: {latest_file}")
        print(f"Checkpoint at {latest_path}")

        api.upload_file(
            path_or_fileobj=latest_path,
            path_in_repo="riff_model_state_dict.ckpt",
            repo_id=repo_id,
        )
        print("Single checkpoint saved.")

    except:
        # see if final state of model exists in current directy and just save that as last resort
        try:
            api.upload_file(
                path_or_fileobj=os.path.join(root_dir, "final_checkpoint.ckpt"),
                path_in_repo="final_checkpoint.ckpt",
                repo_id=repo_id,
            )
            print("final_checkpoint.ckpt successfully saved.")
        finally:
            final_ckpt_path = os.path.join(root_dir, "final_checkpoint.ckpt")
            print(f"Could not find or upload {final_ckpt_path}. Checkpoint saving failed.")