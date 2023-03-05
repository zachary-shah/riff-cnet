## This is the main directory in which riffusion can be trained.

# Instructions:

1. Create processing environment for local computer
**TODO: fix environments**
```conda env create -f envs/processing_env.yml```

2. Run data processing script in processing environment on local computer
```conda activate processing_env```
```sh gen_data.sh```
```conda deactivate```

3. SSH Into VM, which should already have the enviornment set up. Use git pull to pull data into VM from mel-train
```cd mel-train; git pull```

4. Now enter the control enviornment and add control to the model:
(If environment not set up, then first run ```conda env create -f envs/control_env.yml```).
```conda activate control```
```python cnet_riff_add_control.py```
If the model is already present, this will not run, but no worries, we can still move on as this step was already completed. 

5. At this point, a checkpoint of riffusion with cnet layers added should be saved into ./models. Now, we are ready to train: 

*run this for fast training:*
```python cnet_riff_training.py --train_data_dir "train-data/" --batch_size 4 --accumulate_gradient_batches 1```

*if you get CUDA OOM error, run this instead:*
```python cnet_riff_training.py --train_data_dir "train-data/" --batch_size 1 --accumulate_gradient_batches 4```


### NOTES
- Make sure all audio files for training located in "raw-audio" folder. Add "prompt_labels.json" file to root directory to match text prompts to each audio file in raw-audio.
 