
import numpy as np
import os, sys, argparse, json
import cv2
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

import tool_add_control
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

class CnetRiffDataset(Dataset):
    def __init__(self, rootdir):
        self.data = []
        self.rootdir = rootdir
        with open(os.path.join(rootdir, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)
        
        # # Do not forget that OpenCV read images in BGR order.
        source_mod = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target_mod = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # # Normalize source images to [0, 1].
        source_mod = source_mod.astype(np.float32) / 255.0

        # # Normalize target images to [-1, 1].
        target = (target_mod.astype(np.float32) / 127.5) - 1.0

        #TODO: fix normalizations or undo later

        return dict(jpg=target, txt=prompt, hint=source)


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

    parser.add_argument(
        "--add_control",
        type=bool,
        nargs="?",
        default="False",
        help="False loads control model from ckpt without adding control again"
    ) 
    
    
    args = parser.parse_args()
    
    cntrl_riff_path = "./models/control_riffusion_ini.ckpt"

    if args.add_control:
        # get riffusion model downloaded 
        riffusion_path = hf_hub_download(repo_id="riffusion/riffusion-model-v1", filename="riffusion-model-v1.ckpt")
        print(F"Riffusion .ckpt saved to {riffusion_path}")
        # add control to riffusion and save controlled model to cntrl_riff_path
        tool_add_control.tool_add_control(riffusion_path, cntrl_riff_path)

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
    model = create_model('./models/cldm_v15.yaml').cpu()
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