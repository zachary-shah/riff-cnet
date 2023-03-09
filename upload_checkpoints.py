from huggingface_hub import login, HfApi
import os
from stat import S_ISREG, ST_CTIME, ST_MODE

# get file with wget https://raw.githubusercontent.com/zachary-shah/mel-train/main/upload_checkpoints.py

# checkpoint version number
version = 3

# log into hugginface
login()
api = HfApi()
root_dir = os.getcwd()

# upload image logs
print("Uploading images...")
print(os.path.join(root_dir, "image_log/train/"))
api.upload_folder(
    folder_path=os.path.join(root_dir, "image_log/train/"),
    repo_id="zachary-shah/riffusion-cnet",
)
print("Images upoaded.")

try:
    # try uploading all checkpoints first
    print("trying to upload all checkpoints...")
    ckpt_path = os.path.join(root_dir, f"lightning_logs/version_{version}/")
    api.upload_folder(
        folder_path=ckpt_path,
        repo_id="zachary-shah/riffusion-cnet",
    )
    print("All checkpoints uploaded!")

finally:
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

    # or do this: 
    # latest_file = "final_checkpoint.ckpt"
    # latest_path = f"lightning_logs/version_{version}/checkpoints/final_checkpoint.ckpt"

    print(f"Saving checkpoint: {latest_file}")
    print(f"Checkpoint at {latest_path}")

    api.upload_file(
        path_or_fileobj=latest_path,
        path_in_repo="riff_model_state_dict.ckpt",
        repo_id="zachary-shah/riffusion-cnet",
    )

    print("Checkpoint saved!")

try:
    api.upload_file(
        path_or_fileobj=os.path.join(root_dir, "final_checkpoint.ckpt"),
        path_in_repo="final_checkpoint.ckpt",
        repo_id="zachary-shah/riffusion-cnet",
    )
finally:
    final_ckpt_path = os.path.join(root_dir, "final_checkpoint.ckpt")
    print(f"Could not find or upload {final_ckpt_path}")
