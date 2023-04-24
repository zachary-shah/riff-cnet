## DATASET TYPE FOR LOADING INTO TRAINING

import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset

# training dataset built off reference data with the following structure
# <rootdir>: fullfile path that contains:
    # prompt.json --> list of json files in the form {"source": "imgpath", "target": "targetpath", "prompt":, "prompt-str"}
    # source --> folder with canny edge detection spectrograms
    # target --> folder with full audio spectrograms
class CnetRiffDataset(Dataset):
    def __init__(self, rootdir, promptfile="prompt.json"):
        self.data = []
        self.rootdir = rootdir
        self.promptfile = promptfile
        with open(os.path.join(rootdir, promptfile), 'rt') as f:
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

        return dict(jpg=target, txt=prompt, hint=source)
