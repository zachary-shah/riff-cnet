# Controlled Audio Inpainting using Riffusion and ControlNet

## Summary: 

 This project was developed by Zachary Shah, Neelesh Ramachandran, and Mason Wang at Stanford, based on the work done by Seth Forsgren and Hayk Martiros on Riffusion, as well as the work done by Lvmin Zhang and Maneesh Agrawala on ControlNet. Here, we fine-tune Riffusion using the ControlNet architecture in order to create a conditioned audio generation pipeline for audio inpainting. We demonstrate that ControlNet, which has been shown to have immense power in conditional image generation, can also help condition the generation of new audio based on structures inherent to a piece of music represented in a spectrogram of a backtrack or musical accompaniment. 

## Examples of our Model's Samples

To clarify exactly what our model is doing, let's dive into an example. Say we have a song with some vocal lines that we don't like, and we want to generate some new vocals using Stable Diffusion. Let's take this rouglhy 5 second clip of a rock song as our starting point: 

https://user-images.githubusercontent.com/123213526/225162902-84f3feec-6b3c-4020-99ca-cf51c35b0f5b.mp4

You can hear this song has a distinct beat, with a pause near the middle of the clip and then the re-introduction of the prominent guitar feature. Here's what the spectrogram of this clip will look like:

![Generate an up-tempo female vocal rock melody _target](https://user-images.githubusercontent.com/123213526/225163023-fea74ef5-20fb-4a90-b367-02e8b42d4af0.png)

Notice that most of the low portion of the spectrogram details the rhythmic structure of the drum, with some of the mid-frequencies outlining guitar chords with the vocals present as the more curvy lines. You can also see the pause in most of the intstruments, with the singer's last words "Take Me Home" represented as the strong black sinusoidal lines in the mid-frequencies region of the end of the clip. To condition our model to generate this rock song, but with different vocals, we apply a pre-trained CNN called Spleeter to get rid of the vocals from the above clip to isolate just the background, containing the bass, guitar, and audio. Here, you can hear just the background audio:

https://user-images.githubusercontent.com/123213526/225163395-eb17fae0-57ba-4896-8bd8-2a2632b76937.mp4

Now, to condition our model to generate a melody over this background, we turn this background audio into a spectrogram and detect the canny edges of the spectrogram:

![Generate an up-tempo female vocal rock melody _seed](https://user-images.githubusercontent.com/123213526/225163427-d234f0ea-8db2-4e7f-96a8-7c3f216cf3cf.png)

Notice that this outlines many of the edges we saw in the spectrogram of the full audio above, but none of the singer's prominent vocal features are detected due to vocal isolation by Spleeter. Just for fun, if we turn this canny-edge detected spectrogram back into audio, it sounds nothing like the original (warning: turn down your volume):

https://user-images.githubusercontent.com/123213526/225163797-ce5bac66-8fc4-4665-989c-33ff85506463.mp4

Though these canny edges sound like garbage, we can actually use them to condition the diffusion of a song just like the original. Combining this canny edge map with the text prompt to "Generate an up-tempo female vocal rock melody", here is a sample of what our model creates:



Notice that this looks similar to the spectrogram of the full audio, but it is missing some of the key features that pertain to the vocals. 




Here's some more examples:

Here's another 5 second clip rock song: 

https://user-images.githubusercontent.com/123213526/225160377-99585fe0-bbe0-4c63-ad88-511f4f723f20.mp4

Conditioning our model on these edges, with a text prompt to "Generate an uplifting male vocal rock melody", here is some of our model's samples: 

https://user-images.githubusercontent.com/123213526/225161071-d7452104-1f5c-42f2-b4ae-8e9b047c6224.mp4

https://user-images.githubusercontent.com/123213526/225161160-09753807-b3be-4b1c-982d-2f62691d3caa.mp4




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
