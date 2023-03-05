## This is the main directory in which riffusion can be trained.

# Instructions:
(steps 1-2 are already done on the current VM instance. if you wanna try to make new VM instances with different GPU parameters then follow all 3 steps)

1. SSH Into VM, which should already have the enviornment set up. Use git pull to pull data and code into VM from mel-train
```git clone https://github.com/zachary-shah/mel-train.git``` 
```cd mel-train; ls```

2. Now enter the control enviornment and download riffusion / add control to the model:
(If environment not set up, then first run ```conda env create -f envs/control_env.yml```).
```conda activate control```
```python cnet_riff_add_control.py```
If the model is already present, this will not run, but no worries, we can still move on as this step was already completed. 

3. At this point, a checkpoint of riffusion with cnet layers added should be saved into ./models. Now, we are ready to train: 

*run this for fast training:*
```python cnet_riff_training.py --train_data_dir "train-data/" --batch_size 4 --accumulate_gradient_batches 1```

*if you get CUDA OOM error, run this instead:*
```python cnet_riff_training.py --train_data_dir "train-data/" --batch_size 1 --accumulate_gradient_batches 4```

TODO: Figure out why batch_size=1 still doesn't work. need to fix CUDA OOM errors...
