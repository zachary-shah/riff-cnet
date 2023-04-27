"""
SCRIPT FOR PREPROCESSING THE SLAKH-18 DATASET FOR CNET TRAINING
"""

import os
import yaml
import pydub
import numpy as np
import random 
from time import time
import torch

from utils.riffusion_utils import audio_array_to_image
from utils.slakh_utils import get_instrument_str, get_stem_frames, make_slakh_prompt
from cnet_riff_preprocessing import create_prompt_file, append_to_prompt_file, generate_and_save_control

############################################################################################################################################
"""
PARAMETERS (edit this section)
"""
############################################################################################################################################
opt = {}
# for control methods, choose any combination from: "fullspec", "canny", "thresh", "bpf", "sobel", "sobeldenoise"
opt["control_methods"] = ["fullspec", "canny", "sobel", "sobeldenoise"]
# where to load data from 
opt["root_data_dir"] = os.path.join('../', 'babyslakh_16k')
# where to save data to
opt["data_root"] = os.path.join('slakh-preprocessed')
# true to print information about preprocessing as script runs
opt["verbose"] = True
# true to wipe anything present in existing prompt files
opt["new_prompt_files"] = False

# parameters for control generation (if needed)
opt["fs"] = 44100
opt["canny_low_thresh"] = 150
opt["canny_high_thresh"] = 200
opt["denoise_h"] = 15

# parameters for framing
opt["frame_overlap"] = 0 # percentage of frames overlapped, between 0 and 1
opt["frame_len_seconds"] = 5.110 # length of frame, in seconds
opt["frame_min_power_prop"] = 0.4 # minimum ratio between power in frame and power of stem, between 0 and 1
############################################################################################################################################

"""
From a dictionary of numpy arrays of the source and generated stems, make the trianing example desired.
"""
def make_train_example(source_stems, generated_stems, stem_info, song_no, frame_no, ex_no, opt):

    # make target as combo of source and generated stems
    target_stems = source_stems.copy()
    target_stems.update(generated_stems)

    # path naming 
    train_example_name = "s{:04d}_f{:04d}_e{:04d}.jpg".format(song_no, frame_no, ex_no)
    target_path = os.path.join(opt["target_root"], train_example_name)

    # mix target stems and generate/save spectrogram
    target_arr = np.sum(np.array([target_stems[key] for key in target_stems]), axis=0)
    target_spectrogram = audio_array_to_image(target_arr, 
                                  save_img=True,
                                  outpath=target_path[:-4],
                                  sample_rate=opt["fs"],
                                  device=opt["device"],
                                  image_extension="jpg")
    
    # mix source stems and make spectrogram
    source_arr = np.sum(np.array([source_stems[key] for key in source_stems]), axis=0)
    source_spectrogram = audio_array_to_image(source_arr, 
                                  save_img=False,
                                  outpath="",
                                  sample_rate=opt["fs"],
                                  device=opt["device"])
    
    gen_instruments = [stem_info[key]["instrument"] for key in generated_stems]
    src_instruments = [stem_info[key]["instrument"] for key in source_stems]
    prompt = make_slakh_prompt(gen_instruments, src_instruments)

    for control_method in opt["control_methods"]:
        source_path = os.path.join(opt["data_root"], "source-"+control_method, train_example_name)

        # generate control example for each method desired
        generate_and_save_control(source_spectrogram, source_path, control_method, opt)
        
        # add source example to respective prompt file
        append_to_prompt_file(opt["data_root"], [source_path], [target_path], prompt, prompt_filename=f"prompt-{control_method}.json", verbose=False)

    if opt["verbose"]:
        print(f"          {ex_no} - prompt: {prompt}")

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

# create new prompt files if desired
if opt["new_prompt_files"]:
    for control_method in opt["control_methods"]:
        create_prompt_file(opt["data_root"], prompt_filename=f"prompt-{control_method}.json")

# get each song example folder
train_example_dirs = sorted([f for f in os.listdir(opt["root_data_dir"]) if f != '.DS_Store'])

# preprocess each song
for song_no in range(len(train_example_dirs)):

    # load metatdata
    with open(os.path.join(opt["root_data_dir"], train_example_dirs[song_no], 'metadata.yaml'), 'r') as stream:
        metadata = yaml.safe_load(stream)
        
    # get some potentially usefull metadata
    stem_metadata = metadata['stems']
    num_metadata_stems = len(list(stem_metadata.keys()))

    # get only the useful info and from metadata
    stem_info = dict.fromkeys(stem_metadata.keys(), '')
    stems = dict.fromkeys(stem_metadata.keys(), '')
    for stem in stem_metadata:
        try:
            if stem_metadata[stem]['inst_class'] in ["Drums", "Piano", "Bass"]:
                isbgnd = True
            else:
                isbgnd = False

            # update text description for some things
            stem_info[stem] = {"class": stem_metadata[stem]['inst_class'],
                            "instrument": get_instrument_str(stem_metadata[stem]),
                            "background": isbgnd} 
            
            # load each stem as a pybud audio file
            stems[stem] = pydub.AudioSegment.from_file(os.path.join(opt["root_data_dir"], train_example_dirs[song_no], metadata['audio_dir'], f"{stem}.wav"))
            # update frame rate if needed
            if stems[stem].frame_rate != opt["fs"]:
                stems[stem] = stems[stem].set_frame_rate(opt["fs"])
        except:
            stem_info.pop(stem)
            stems.pop(stem)

    num_actual_stems = len(list(stems.keys()))

    if opt["verbose"]:
        print(f"PROCESSING SONG {song_no+1}/{len(train_example_dirs)}: ")
        print(f"  stems in metadata: {num_metadata_stems}")
        print(f"  loaded stems: {num_actual_stems}")
        print()

    # get frames in audio where each stem is present
    frames = dict.fromkeys(stems.keys(), "")
    stem_names = list(stems.keys())
    for stem_name in stem_names:
        segment = stems[stem_name]

        frames[stem_name] = get_stem_frames(segment, 
                                            overlap = opt["frame_overlap"],
                                            frame_seconds = opt["frame_len_seconds"],
                                            min_power_prop = opt["frame_min_power_prop"],
                                            fs = opt["fs"])

    # get list of all valid frames that exist
    frame_nos = sorted(list(set([itm for l in [list(frames[k].keys()) for k in frames] for itm in l])))
    num_frames = len(frame_nos)

    # make frame number the keys for all frames
    stem_arrs_by_frame = dict.fromkeys(frame_nos,"")
    for stem in frames:
        for arr in frames[stem]:
            if stem_arrs_by_frame[arr] == "":
                stem_arrs_by_frame[arr] = {}
            stem_arrs_by_frame[arr][stem] = frames[stem][arr]

    # get stems that exist in each frame
    stem_names_by_frame = {}
    for frame_no in frame_nos:
        frame_stem_names = []
        for stem_name in frames:
            if frame_no in list(frames[stem_name].keys()):
                frame_stem_names.append(stem_name)
        stem_names_by_frame[frame_no] = frame_stem_names

    if opt["verbose"]:
        print(f"  number of frames: {num_frames}")
        print()

    ## ITERATE OVER FRAMES AND BUILD TRAINING EXAMPLES FOR EACH FRAME
    for frame_no in frame_nos:

        if opt["verbose"]:
            print(f"  FRAME {1+frame_no}/{len(frame_nos)}:")

        # COMBONATORIAL SET MAKING RULES (at most 22 examples procured per frame)
        # types of sets:
          # if at least 2 background stems and 1 non-background stem is present: 
              # make at most 3 combinations of generating background stems from a non-background stem
              # make at most 3 combinations of generating a non-background stems from background stems
          # if at least 4 stems present total:
              # make at most 8 examples of generating 1 stem from all other stems
          # if at least 5 stems present total:
              # make at most 4 examples of generating 2 stems from all other stems
        
        # get stems that are considered background
        stems_in_frame = np.array(stem_names_by_frame[frame_no])
        bgnd_stems = stems_in_frame[np.array([stem_info[key]["background"] for key in stems_in_frame])]
        non_bgnd_stems = stems_in_frame[np.array([not stem_info[key]["background"] for key in stems_in_frame])]

        ex_no = 0
        # make at least one melody from bgnd and bgnd from melody examples, if there is enough of each present in frame
        if len(bgnd_stems) >= 2 and len(non_bgnd_stems) >= 1: 
            # get at most 3 examples where melody/harmony are generated from background
            source_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in bgnd_stems])
            generated_stem_names = random.sample(list(non_bgnd_stems), min(3, len(non_bgnd_stems)))
            generated_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in generated_stem_names])
            if opt["verbose"]: 
                print("    Generating melodies from bgnd: ")
            for generated_stem in generated_stems:
                
                ex_no = make_train_example(source_stems, {generated_stem:generated_stems[generated_stem]}, stem_info, song_no, frame_no, ex_no, opt)

            # get at most 3 examples where background generated from melody/harmony
            generated_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in bgnd_stems])
            source_stems_name = random.sample(list(non_bgnd_stems), min(3, len(non_bgnd_stems)))
            if opt["verbose"]: 
                print("    Generating bgnds from a melody: ")
            for source_stem_name in source_stems_name:
                source_stem = dict({source_stem_name: stem_arrs_by_frame[frame_no][source_stem_name]})
                ex_no = make_train_example(source_stem, generated_stems, stem_info, song_no, frame_no, ex_no, opt)

        # make all combinations of 1 generated stem, with max 6 generated stems, if 4 or more stems
        if len(stems_in_frame) > 4:
            # at most 8 examples
            generable_stems = random.sample(list(stems_in_frame), np.min([len(stems_in_frame), 8]))
            if opt["verbose"]: 
                print(f"    Making {len(generable_stems)} examples out of {len(stems_in_frame)} stems in frame: ")
            for gen_stem in generable_stems:
                generated_stem = dict({gen_stem: stem_arrs_by_frame[frame_no][gen_stem]})
                source_stems_name = stems_in_frame[stems_in_frame != gen_stem]
                source_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in source_stems_name])
                ex_no = make_train_example(source_stems, generated_stem, stem_info, song_no, frame_no, ex_no, opt)

        # make some combinations of 2 generated stem, with max 4 examples, if 5 or more stems
        if len(stems_in_frame) > 5:
            # at most 4 examples
            generable_stems = random.sample(list(stems_in_frame), np.min([len(stems_in_frame), 5]))
            if opt["verbose"]: 
                print(f"    Making {len(generable_stems)-1} examples out of {len(stems_in_frame)} stems in frame: ")
            for i in range(len(generable_stems) - 1):
                generated_stems = dict({generable_stems[i]: stem_arrs_by_frame[frame_no][generable_stems[i]],
                                    generable_stems[i+1]: stem_arrs_by_frame[frame_no][generable_stems[i+1]]})
                source_stems_name = stems_in_frame[(stems_in_frame != generable_stems[i]) * (stems_in_frame != generable_stems[i+1])]
                source_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in source_stems_name])
                ex_no = make_train_example(source_stems, generated_stems, stem_info, song_no, frame_no, ex_no, opt)

        if opt["verbose"]: 
            print(f"    frame no {frame_no} made {ex_no} frames")
        num_examples_total += ex_no

# script information
time_elapsed = (time() - time_start) / 60
print("Preprocessing complete! Summary:")
print(f" - preprocessed {len(train_example_dirs)} songs")
print(f" - generated {num_examples_total} examples total")
print(" - control methods:", opt["control_methods"])
print(f" - runtime: {time_elapsed} min")