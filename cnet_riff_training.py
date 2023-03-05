
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cnet_riff_dataset import CnetRiffDataset

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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
        "--sd_locked",
        type=bool,
        nargs="?",
        default=True,
        help="False to unlock part of model that might make it easier to learn unique image types, but risk corrupting model weights."
    ) 
    parser.add_argument(
        "--logger_freq",
        type=int,
        nargs="?",
        default=300,
        help="Step Frequency for logger to save images."
    ) 
    parser.add_argument(
        "--only_mid_control",
        type=int,
        nargs="?",
        default=False,
        help="True to limit control to only the middle layers of model."
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
        help="If batch size = 1, then make accumulate_gradient_batches = 4."
    )
    args = parser.parse_args()

    # Configs
    cntrl_riff_path = "./models/control_riffusion_ini.ckpt"
    logger_freq = args.logger_freq
    only_mid_control = args.only_mid_control
    batch_size = args.batch_size
    train_data_dir = args.train_data_dir
    accumulate_gradient_batches = args.accumulate_gradient_batches

    # DEFAULT IS TRUE. but reccomend trying false for unique image types. but then lower LR to 2e-6
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

    # load in dataset
    dataset = CnetRiffDataset(train_data_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    # make logger and model trainer
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], accumulate_grad_batches=accumulate_gradient_batches)

    # Train!
    trainer.fit(model, dataloader)

    # TODO: make sure checkpoints being saved??

if __name__ ==  '__main__':
    main()