"""
SCRIPT FOR PREPROCESSING THE GTZAN DATASET FOR CNET TRAINING
"""
import os, json
import numpy as np
from pathlib import Path
from time import time
import torch

from utils.audio_segment_utils import segment_audio
from utils.riffusion_utils import audio_array_to_image
from cnet_riff_preprocessing import append_to_prompt_file, generate_and_save_control
from utils import spleeter_utils

############################################################################################################################################
"""
PARAMETERS (edit this section)
"""
############################################################################################################################################
opt = {}
# for control methods, choose any combination from: "fullspec", "canny", "thresh", "bpf", "sobel", "sobeldenoise"
opt["control_methods"] = ["fullspec", "canny", "sobel", "sobeldenoise"]
# where to load data from 
opt["root_data_dir"] = os.path.join('../','gtzan')
opt["raw_audio_dir"] = os.path.join(opt["root_data_dir"],'raw-audio')
opt["prompt_labels_filepath"] = os.path.join(opt["root_data_dir"],'prompt_labels.json')

# where to save data to
opt["data_root"] = os.path.join('gtzan-preprocessed')
# true to print information about preprocessing as script runs
opt["verbose"] = True

# parameters for control generation (if needed)
opt["fs"] = 44100
opt["canny_low_thresh"] = 150
opt["canny_high_thresh"] = 200
opt["denoise_h"] = 15
############################################################################################################################################

"""
From a dictionary of numpy arrays of the source and generated stems, make the training example desired.
"""
def make_train_example(source_arr, target_arr, prompt, audio_filename, ex_no, opt):

    # path naming 
    train_example_name = f'{audio_filename}_e{ex_no}.jpg'
    target_path = os.path.join(opt["target_root"], train_example_name)

    # mix target stems and generate/save spectrogram
    target_spectrogram = audio_array_to_image(target_arr, 
                                  save_img=True,
                                  outpath=target_path[:-4],
                                  sample_rate=opt["fs"],
                                  device=opt["device"],
                                  image_extension="jpg")
    
    # mix source stems and make spectrogram
    source_spectrogram = audio_array_to_image(source_arr, 
                                  save_img=False,
                                  outpath="",
                                  sample_rate=opt["fs"],
                                  device=opt["device"])

    for control_method in opt["control_methods"]:
        source_path = os.path.join(opt["data_root"], "source-"+control_method, train_example_name)
        # generate control example for each method desired
        generate_and_save_control(source_spectrogram, source_path, control_method, opt)
        # add source example to respective prompt file
        append_to_prompt_file(opt["data_root"], [source_path], [target_path], prompt, prompt_filename=f"prompt-{control_method}.json", verbose=False)

    if opt["verbose"]:
        print(f"     {ex_no} - prompt: {prompt}")
    ex_no += 1
    return ex_no

# tracking
num_examples_total = 0
time_start = time()

# cuda if possible
if torch.cuda.is_available():
    opt["device"] = "cuda"
else:
    opt["device"] = "cpu"

# control/target data folders
opt["control_roots"] = [os.path.join(opt["data_root"], "source-"+mthd) for mthd in opt["control_methods"]]
opt["target_root"] = os.path.join(opt["data_root"], 'target')

# make all directories needed
os.makedirs(opt["data_root"], exist_ok=True)
for control_root in opt["control_roots"]:
    os.makedirs(control_root, exist_ok=True)
os.makedirs(opt["target_root"], exist_ok=True)

# get all data examples
audio_files = os.listdir(opt["raw_audio_dir"])

# get all prompts in prompt_file as dictionary
prompt_dict = {}
p_count = 0
with open(opt["prompt_labels_filepath"], 'r') as prompt_file:
    for line in prompt_file:
        data = json.loads(line)
        prompt_dict[data['file']] = data['prompt']
        p_count += 1
if opt["verbose"]: print(f"Read {p_count} prompts from prompt_file.json.")

# iterate through all audio file examples
for (num_file, audio_file) in enumerate(audio_files):
        
        if opt["verbose"]:
            print(f"AUDIO FILE {num_file+1}/{len(audio_files)}:")

        audio_filename = audio_file[:audio_file.index(".wav")]

        # audio splitting
        splits = spleeter_utils.separate_audio(os.path.join(opt["raw_audio_dir"], audio_file), fs=opt["fs"], stem_num=2)
        accompaniment_audio = splits['accompaniment']
        full_audio = splits['full_audio']
        vocal_audio = splits['vocals']

        # get audio segments with pitch augmentation on (should be 72 segments total)
        full_audio_segments = segment_audio(full_audio, fs=opt["fs"], num_segments=5, pitch_augment=True)
        accompaniment_audio_segments = segment_audio(accompaniment_audio, fs=opt["fs"], num_segments=5, pitch_augment=True)
        vocal_audio_segments = segment_audio(vocal_audio, fs=opt["fs"], num_segments=5, pitch_augment=True)

        # remove segments with low vocal power
        acceptable_inds = []
        for i, accompaniment_audio_segment in enumerate(accompaniment_audio_segments):
                if np.linalg.norm(vocal_audio_segments[i]) > np.linalg.norm(accompaniment_audio_segment)*0.1:
                    acceptable_inds.append(i)
                else:
                    if opt["verbose"]: print("    Vocals not detected in segement " + str(i))
        full_audio_segments = full_audio_segments[acceptable_inds]
        accompaniment_audio_segments = accompaniment_audio_segments[acceptable_inds]
        vocal_audio_segments = vocal_audio_segments[acceptable_inds]

        if opt["verbose"]:
            print(f"  Total number of segments for {audio_filename}: {full_audio_segments.shape[0]}")

        # get prompt
        if audio_file in prompt_dict:
            song_prompt = prompt_dict[audio_file]
        else:
            song_prompt = "Generate a vocal melody."

        # make training example for each frame
        ex_no = 0
        for i in range(len(full_audio_segments)):

            #10% of the time, also say given background
            if np.random.rand() < 0.1:
                 prompt = "Given background audio: " + song_prompt
            else:
                 prompt = song_prompt

            # generate vocal melody from background
            ex_no = make_train_example(source_arr=accompaniment_audio_segments[i],
                                target_arr=full_audio_segments[i],
                                prompt=prompt,
                                audio_filename=audio_filename,
                                ex_no=ex_no,
                                opt=opt)
             
        num_examples_total += ex_no

# script information
time_elapsed = (time() - time_start) / 60
print("Preprocessing complete! Summary:")
print(f" - preprocessed {len(audio_files)} songs")
print(f" - generated {num_examples_total} examples total")
print(" - control methods:", opt["control_methods"])
print(f" - runtime: {time_elapsed} min")