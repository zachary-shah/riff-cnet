# Controlled Audio Inpainting using Riffusion and ControlNet

## Summary: 

 This project was developed by Zachary Shah, Neelesh Ramachandran, and Mason Wang at Stanford, based on the work done by Seth Forsgren and Hayk Martiros on Riffusion, as well as the work done by Lvmin Zhang and Maneesh Agrawala on ControlNet. Here, we fine-tune Riffusion using the ControlNet architecture in order to create a conditioned audio generation pipeline for audio inpainting. We demonstrate that ControlNet, which has been shown to have immense power in conditional image generation, can also help condition the generation of new audio based on structures inherent to a piece of music represented in a spectrogram of a backtrack or musical accompaniment. 

## Here are some examples of our model at work: 

<TODO: ADD>


## To use on our pre-trained model: 

Our pretrained model is located at our HuggingFace repo. Access this model using:

```git install lfs```
```git clone https://huggingface.co/zachary-shah/riffusion-cnet-v2```

See how to use our pre-trained model using the Jupyter Notebook ```sample_riff_cnet.ipynb```. This script is best run in Google Colab. To run, make sure to set up the necessary Python environment at ```conda env create -f envs/control_env.yml```, and clone this repo in your Google Drive using ```git clone https://github.com/zachary-shah/mel-train.git```. Run ```sample_riff_cnet.ipynb``` within the cloned repo.


## To train Riffusion using ControlNet from scratch:

Note that this is a compute-intensive process and requires the use of at least 25 GiB GPU vRAM.  

1. From terminal, clone this repo to pull in the data and code:

```git clone https://github.com/zachary-shah/mel-train.git``` 

```cd mel-train; ls```

2. (Optional) We provide all our preprocessed train data in the ```train-data/``` directory, which is generated from the raw audio files in ```raw-audio```. To re-generate this data, first set up the processing environment ```conda env create -f envs/processing_env.yml```, and then simply run:

```gen_data.sh```

Otherwise, if you would like to use custom data to train this model, you can apply our processing pipeline to prepare training data from a set of .wav audio files. Your audio files must be located in the folder ```<your_audio_files_directory>```, and each audio file must have a corresponding prompt entered into ```<your_prompt_file>.json```. See ```prompt_file.json``` for reference on how to set this file up. Then run this script:

```python make_cnet_dataset.py --audio_dir "<your_audio_files_directory>" --train_data_dir "<output_train_data_directory>" --prompt_file "<your_prompt_file>"```

*Note 1:* preprocessing requires the use of Spleeter to separate the background audio from vocals, but we observe that different Python versions and packages were required depending on the user's operating system. You may need to play around with different versions of Python to get this function to work. 

*Note 2:* for some reason, there is a bug where only 5 audio files can be processed at a time. See ```gen_data.sh``` for a solution to this, which involves limiting the above script to processing 5 files at a time, and cycles through starting at every 5 indexes for processing all the data at once.

3. Now enter the control enviornment and download riffusion to add control to the model:

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

If training is extremely slow, can also lower max_steps to 10000, as we saw convergence happen even at this point. Note that this training script is also set up to run for a maximum of 12 hours. To change this max training time, manually go into cnet_riff_training.py and change the ```max_train_time``` variable as desired.

4. After training completes, run the following script to upload the model to HuggingFace if desired, with the version of the image logger provided as the first argument and the repo to upload to as the second (using my repo as an example):

```python upload_checkpoints.py 3 zachary-shah/riffusion-cnet-v2```
