
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
        default=10000,
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
    args = parser.parse_args()

    # Configs
    cntrl_riff_path = "./models/control_riffusion_ini.ckpt"
    max_train_time = "00:06:00:00" # max training time is 6 hours (form is "DD:HH:MM:SS")
    logger_freq = args.logger_freq
    train_data_dir = args.train_data_dir
    accumulate_gradient_batches = args.accumulate_gradient_batches
    max_steps = args.max_steps
    save_gpu_memory = args.save_gpu_memory
    only_mid_control = args.only_mid_control
    batch_size = args.batch_size
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.max_split_size}"

    # SD_LOCKED: default is True. but reccomend trying false for unique image types. but then lower LR to 2e-6
    if args.sd_locked:
        sd_locked = True 
        learning_rate = 1e-5
    else:
        sd_locked = False
        learning_rate = 2e-6

    # load dataset. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(cntrl_riff_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # changes to make to save gpu memory
    if save_gpu_memory:
        move_metrics_to_cpu = True # reccomended to decrease gpu load, but makes training slower
        accumulate_gradient_batches = batch_size
        batch_size = 1 # accumulate gradients rather than train in batches. also slows training
        precision = 16 # divide 32-precision by 1/2
    else:
        move_metrics_to_cpu = False
        precision = 16 # TODO: maybe change this to 32 if things are successful

    # load in dataset
    dataset = CnetRiffDataset(train_data_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    print(f"Number of epochs to train: {np.ceil(max_steps * batch_size / len(dataset))}")

    # make logger and model trainer
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, 
                         precision=precision, 
                         callbacks=[logger], 
                         accumulate_grad_batches=accumulate_gradient_batches, 
                         max_steps=max_steps, 
                         move_metrics_to_cpu=move_metrics_to_cpu,
                         max_time=max_train_time)

    # Train!
    trainer.fit(model, dataloader)
    print("Training complete. Models and logging saved to directory.")

if __name__ ==  '__main__':
    main()