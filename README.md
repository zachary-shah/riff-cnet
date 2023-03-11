## This is the main directory in which riffusion can be trained.

# Instructions for use:

1. From terminal, clone this repo to pull in the data and code:

```git clone https://github.com/zachary-shah/mel-train.git``` 

```cd mel-train; ls```

2. Now enter the control enviornment and download riffusion / add control to the model:

```conda env create -f envs/control_env.yml```

```conda activate control```

```python cnet_riff_add_control.py```


3. At this point, a checkpoint of riffusion with the added ControlNet layers should be saved into ./models. Now, we are ready to train: 

*run this for fast training:*

```python cnet_riff_training.py --train_data_dir "train-data/" --max_steps 100000```

*if you get CUDA OOM error, run this instead:*

```python cnet_riff_training.py --train_data_dir "train-data/" --max_steps 100000 --save_gpu_memory True```

*if CUDA OOM error persists, try this (lower batch size to 1):*

```python cnet_riff_training.py --train_data_dir "train-data/" --max_steps 100000 --batch_size 1 --save_gpu_memory True --only_mid_control True```

If training is extremely slow, can also lower max_steps to 5000 or even 1000. Note that this training script is also set up to run for a maximum of 6 hours. To change this max training time, manually go into cnet_riff_training.py and change the ```max_train_time``` variable as desired.
