
import numpy as np
import librosa
import os, sys, argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from ControlNet import tool_add_control
from ControlNet.cldm.logger import ImageLogger
from ControlNet.cldm.model import create_model, load_state_dict

from cnet_riff_dataset import CnetRiffDataset

def main():
    # names of files to segment
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data_dir",
        type=str,
        nargs="?",
        default="train-data/",
        help="directory to read training dataset from"
    )

    parser.add_argument(
        "--sd_locked",
        type=bool,
        nargs="?",
        default="True",
        help="False to unlock part of model that might make it easier to learn unique image types, but risk corrupting model weights."
    ) 
    
    args = parser.parse_args()

    # get riffusion model downloaded 
    riffusion_path = hf_hub_download(repo_id="riffusion/riffusion-model-v1", filename="riffusion-model-v1.ckpt")

    # add control to riffusion and save controlled model to cntrl_riff_path
    cntrl_riff_path = "./models/control_riffusion_ini.ckpt"
    tool_add_control(riffusion_path, cntrl_riff_path)

    # Configs
    batch_size = 4
    logger_freq = 300
    only_mid_control = False

    # DEFAULT IS TRUE. but reccomend trying false for unique image types. but then lower LR to 2e-6
    if args.sd_locked:
        sd_locked = True 
        learning_rate = 1e-5
    else:
        sd_locked = False
        learning_rate = 2e-6

    # load dataset. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(cntrl_riff_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # load in dataset
    dataset = CnetRiffDataset(args.train_data_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)

if __name__ ==  '__main__':
    main()