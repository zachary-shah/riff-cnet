
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cnet_riff_dataset import CnetRiffDataset

import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_dir",
        type=str,
        nargs="?",
        default="train-data/",
        help="directory to read training dataset from"
    )
    parser.add_argument(
        "--logger_freq",
        type=int,
        nargs="?",
        default=500,
        help="Step Frequency for logger to save images."
    ) 
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        default=4,
        help="Batch size for dataloader. 4 is default, but lower to 1 if encountering CUDA OOM issues."
    )
    parser.add_argument(
        "--accumulate_gradient_batches",
        type=int,
        nargs="?",
        default=1,
        help="If batch size = 1, then make accumulate_gradient_batches 2 or 4."
    )
    parser.add_argument(
        "--max_split_size",
        type=int,
        nargs="?",
        default=512,
        help="Max partition size in GB for cuda. Lower default (e.g. to 128GB) to save gpu memory for cuda splits."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        nargs="?",
        default=100000,
        help="Max number of steps for training."
    )
    parser.add_argument(
        "--save_gpu_memory",
        type=bool,
        nargs="?",
        default=False,
        help="True to make changes to code to save gpu memory."
    )
    parser.add_argument(
        "--sd_locked",
        type=bool,
        nargs="?",
        default=True,
        help="False to unlock part of model that might make it easier to learn unique image types, but risk corrupting model weights."
    ) 
    parser.add_argument(
        "--only_mid_control",
        type=bool,
        nargs="?",
        default=False,
        help="True to limit control to only the middle layers of model."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        nargs="?",
        default="",
        help="For denoting control method used, if desired, for file access and saving."
    )
    parser.add_argument(
        "--max_train_time",
        type=str,
        nargs="?",
        default="00:12:00:00",
        help="max training time default is 12 hours (form is \"DD:HH:MM:SS\")"
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        nargs="?",
        default="./models/cldm_v15.yaml",
        help="max training time default is 12 hours (form is \"DD:HH:MM:SS\")"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        nargs="?",
        default=2,
        help="number of workers for the dataloader"
    )
    args = parser.parse_args()

    # Unchangeable configs
    cntrl_riff_path = "./models/control_riffusion_ini.ckpt"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.max_split_size}"

    # SD_LOCKED: default is True. but reccomend trying false for unique image types. but then lower LR to 2e-6
    if args.sd_locked:
        learning_rate = 1e-5
    else:
        learning_rate = 2e-6

    # load dataset. Pytorch Lightning will automatically move it to GPUs.
    # make models folder if not already made
    os.makedirs("models/", exist_ok=True)

    # where final checkpoint will be saved to
    if len(args.control_method) >= 1:
        save_path = "riff-cnet-"+args.control_method+"-final.ckpt"
    else:
        save_path = "riff-cnet-final.ckpt"
    final_ckpt_path = os.path.join("models/",save_path)

    # create controlnet and load in riffusion weights
    model = create_model(args.model_config_path).cpu()
    model.load_state_dict(load_state_dict(cntrl_riff_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    # changes to make to save gpu memory
    if args.save_gpu_memory:
        move_metrics_to_cpu = True # reccomended to decrease gpu load, but makes training slower
        precision = 16 # divide 32-precision by 1/2
    else:
        move_metrics_to_cpu = False
        precision = 32 # TODO: maybe change this to 32 if things are successful

    # load in dataset
    if len(args.control_method) >= 1:
        dataset = CnetRiffDataset(args.train_data_dir, promptfile="prompt-"+args.control_method+".json")
    else:
        dataset = CnetRiffDataset(args.train_data_dir)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    print(f"Number of epochs to train: {np.ceil(args.max_steps * args.batch_size / len(dataset))}")

    # make logger and model trainer
    logger = ImageLogger(batch_frequency=args.logger_freq, clamp=False, save_dir_root="image_log_"+args.control_method)
    trainer = pl.Trainer(gpus=1, 
                         precision=precision, 
                         callbacks=[logger], 
                         accumulate_grad_batches=args.accumulate_gradient_batches, 
                         max_steps=args.max_steps, 
                         move_metrics_to_cpu=move_metrics_to_cpu,
                         max_time=args.max_train_time)

    # Train!
    trainer.fit(model, dataloader)
    print("Training complete. Saving final checkpoint...")

    # SAVE FINAL CHECKPOINT
    trainer.save_checkpoint(final_ckpt_path)

    print(f"Final checkpoint saved to {final_ckpt_path}!")


if __name__ ==  '__main__':
    main()