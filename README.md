## This is the main directory in which riffusion can be trained.

### Instructions:
Simply run this line in terminal:

```sh main.sh```

This will create the necessary environments to process the raw audio (processing_env) and to train the ControlNet controlled-riffusion model (control_env). 

Exact .yml files for these environments still need to be updated. 

### NOTES

- Make sure all audio files for training located in "raw-audio" folder. Add "prompt_labels.json" file to root directory to match text prompts to each audio file in raw-audio.
