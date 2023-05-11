f"""
SCRIPT FOR PREPROCESSING THE SLAKH-18 DATASET FOR CNET TRAINING
Terminology:
    - stem name: Distortion Guitar
    - stem ID: S00
    - stem info: {'S00', 'array...', 'dtype...'}
"""

import os
import yaml
import pydub
import numpy as np
import random 
from time import time
import torch
import librosa

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
opt["data_root"] = os.path.join('slakh-preprocessed-narrow')
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
Note: complexity arises from non-uniqueness of mapping between ID and name,
along with potential nonuniformity across different songs
(As users, we want to filter for just 'Harmonica', which might be S00
for one track, S01 for another.)"""
MODE_INSTR_NAME_MAP = {
    0: "", # Allow any and all instruments to be generated (widest task)
    1: ["Piano", "Grand Piano", "Acoustic Grand Piano", "Vibraphone", "Clavinet", "Saxophone", "Flute", "Harmonica"], # Category: generate melodic instruments only
    2: ["Percussive Organ"], # Category: custom subset
}

DESIRED_MODE = 1

"""
Given a set of generated instrument stems, perform some narrowing-down of the list to enable
better training on the narrower task.
"""
def get_narrow_gen_stem_IDs(gen_instrument_stem_IDs, stem_ID_to_name, mode_num=1):
    if mode_num == 0:
        return gen_instrument_stem_IDs # No changes/narrowing needs to be done
    
    valid_gen_instrument_names = MODE_INSTR_NAME_MAP[mode_num]
    out_gen_instrument_IDs = []
    for id in gen_instrument_stem_IDs:
        if stem_ID_to_name[id] in valid_gen_instrument_names:
            out_gen_instrument_IDs.append(id)
            
    return out_gen_instrument_IDs

"""
From a dictionary of numpy arrays of the source and generated stems, make the training example desired.
"""
def make_train_example(source_stems_info, generated_stems_info, all_stem_info, song_no, frame_no, ex_no, opt):

    # Initialize target as combo of source and generated stems
    target_stems_infos = source_stems_info.copy()
    target_stems_infos.update(generated_stems_info)

    # Path naming 
    train_example_name = "s{:04d}_f{:04d}_e{:04d}.jpg".format(song_no, frame_no, ex_no)
    target_path = os.path.join(opt["target_root"], train_example_name)

    # Mix target stems and generate / save spectrogram for external use
    target_arr = np.sum(np.array([target_stems_infos[id] for id in target_stems_infos]), axis=0)
    _ = audio_array_to_image(target_arr, 
                            save_img=True,
                            outpath=target_path[:-4],
                            sample_rate=opt["fs"],
                            device=opt["device"],
                            image_extension="jpg")
    
    # Mix source stems and make spectrogram
    source_arr = np.sum(np.array([source_stems[id] for id in source_stems]), axis=0)
    source_spectrogram = audio_array_to_image(source_arr, 
                                  save_img=False,
                                  outpath="",
                                  sample_rate=opt["fs"],
                                  device=opt["device"])
    
    gen_instrument_names = [all_stem_info[id]["instrument"] for id in generated_stems_info.keys()]
    src_instrument_names = [all_stem_info[id]["instrument"] for id in source_stems_info.keys()]
    
    prompt = make_slakh_prompt(gen_instrument_names, src_instrument_names)

    for control_method in opt["control_methods"]:
        source_path = os.path.join(opt["data_root"], "source-"+control_method, train_example_name)

        # generate control example for each method desired
        generate_and_save_control(source_spectrogram, source_path, control_method, opt)
        
        # add source example to respective prompt file
        append_to_prompt_file(opt["data_root"], [source_path], [target_path], prompt, prompt_filename=f"prompt-{control_method}.json", verbose=False)

    if opt["verbose"]:
        print(f"\t\t{ex_no} - prompt: {prompt}")

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

# Filter by song validity (determined here via beats-per-minute in dynamic sense)
valid_song_nos = []
MIN_BPM = 40
MAX_BPM = 140
desired_percent = 60 # Keep song if at least this percent meets desired tempo metrics
for song_no, folder in enumerate(train_example_dirs):
    mix_audio, audio_sr = librosa.load(os.path.join(opt["root_data_dir"], folder, 'mix.wav'), duration=60)
    onset_env = librosa.onset.onset_strength(mix_audio, sr=audio_sr)
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=audio_sr, aggregate=None)
    valid_idxs = np.where((dtempo >= MIN_BPM) & (dtempo <= MAX_BPM), 1, 0)
    valid_bpm = sum(valid_idxs) / len(dtempo) * 100 >= desired_percent
    
    if valid_bpm:
        valid_song_nos.append(song_no)

print(f"Valid Tempo Songs: {np.add(np.array(valid_song_nos), 1)}", flush=True)

# preprocess each song
for song_no, song in enumerate(train_example_dirs):
    
    if not song_no in valid_song_nos:
        print(f"INFO: Skipping song {song_no} due to filtering constraints")
        continue

    # load metadata
    with open(os.path.join(opt["root_data_dir"], song, 'metadata.yaml'), 'r') as stream:
        metadata = yaml.safe_load(stream)
        
    # get some potentially useful metadata
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
            stems[stem] = pydub.AudioSegment.from_file(os.path.join(opt["root_data_dir"], song, metadata['audio_dir'], f"{stem}.wav"))
            
            # update frame rate if needed
            if stems[stem].frame_rate != opt["fs"]:
                stems[stem] = stems[stem].set_frame_rate(opt["fs"])
        except:
            stem_info.pop(stem)
            stems.pop(stem)


    # Each song has a potentially different map of stem IDs to stem names
    stem_IDs = list(stems.keys())
    num_actual_stems = len(stem_IDs)
    stem_names = [stem_metadata[key]['midi_program_name'] for key in list(stem_metadata.keys())]
    
    # Dict approach below doesn't work cleanly due to non-uniqueness issues
    # but can be made to work, in a sense
    stem_ID_to_name = {}
    for stem_ID in stem_IDs:
        curr_name = stem_metadata[stem_ID]['midi_program_name']
        stem_ID_to_name[stem_ID] = curr_name

    if opt["verbose"]:
        print(f"PROCESSING SONG {song_no+1}/{len(train_example_dirs)}: ")
        print(f"\tStem names for song: {', '.join(stem_names)}\n")
        print(f"\tNum stems in metadata: {num_metadata_stems}")
        print(f"\tNum loaded stems: {num_actual_stems}\n")

    # get frames in audio where each stem is present
    frames = dict.fromkeys(stems.keys(), "")
    for stem_ID in stem_IDs:
        segment = stems[stem_ID]

        frames[stem_ID] = get_stem_frames(segment, 
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
    stem_IDs_by_frame = {}
    for frame_no in frame_nos:
        frame_stem_IDs = []
        for stem_ID in frames:
            if frame_no in list(frames[stem_ID].keys()):
                frame_stem_IDs.append(stem_ID)
        stem_IDs_by_frame[frame_no] = frame_stem_IDs

    if opt["verbose"]:
        print(f"\tnumber of frames: {num_frames}\n")

    ## ITERATE OVER FRAMES AND BUILD TRAINING EXAMPLES FOR EACH FRAME
    for frame_no in frame_nos:

        if opt["verbose"]:
            print(f"\tFRAME {frame_no+1}/{len(frame_nos)}:", flush=True) # Flush output buffer on each frame

        # COMBINATORIAL SET MAKING RULES (at most 22 examples procured per frame)
        # types of sets:
          # if at least 2 background stems and 1 non-background stem is present: 
              # make at most 3 combinations of generating background stems from a non-background stem
              # make at most 3 combinations of generating a non-background stems from background stems
          # if at least 4 stems present total:
              # make at most 8 examples of generating 1 stem from all other stems
          # if at least 5 stems present total:
              # make at most 4 examples of generating 2 stems from all other stems
        
        # get stems that are considered background
        stem_IDs_in_frame = np.array(stem_IDs_by_frame[frame_no])
        bgnd_stem_IDs = stem_IDs_in_frame[np.array([stem_info[key]["background"] for key in stem_IDs_in_frame])]
        non_bgnd_stem_IDs = stem_IDs_in_frame[np.array([not stem_info[key]["background"] for key in stem_IDs_in_frame])]
        
        valid_gen_non_bgnd_stem_IDs = get_narrow_gen_stem_IDs(non_bgnd_stem_IDs, stem_ID_to_name, mode_num=DESIRED_MODE)
        
        ex_no = 0
        # make at least one melody from bgnd and bgnd from melody examples, if there is enough of each present in frame
        # in narrower version: only generate melody/harmony from background
        if len(bgnd_stem_IDs) >= 2 and len(valid_gen_non_bgnd_stem_IDs) >= 1: 
            # get at most 3 examples where melody/harmony are generated from background
            source_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in bgnd_stem_IDs])
            generated_stem_IDs = random.sample(list(valid_gen_non_bgnd_stem_IDs), min(3, len(valid_gen_non_bgnd_stem_IDs)))
            
            generated_stem_infos = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in generated_stem_IDs])
            if opt["verbose"]:
                print("\t\tGenerating melodies from bgnd: ")
            for generated_stem_info in generated_stem_infos:
                gen_stem_set = {generated_stem_info:generated_stem_infos[generated_stem_info]}
                ex_no = make_train_example(source_stems, gen_stem_set, stem_info, song_no, frame_no, ex_no, opt)

            # get at most 3 examples where background generated from melody/harmony
            # generated_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in bgnd_stems])
            # source_stems_name = random.sample(list(non_bgnd_stems), min(3, len(non_bgnd_stems)))
            # if opt["verbose"]: 
            #     print("\tGenerating bgnds from a melody: ")
            # for source_stem_name in source_stems_name:
            #     source_stem = dict({source_stem_name: stem_arrs_by_frame[frame_no][source_stem_name]})
            #     ex_no = make_train_example(source_stem, generated_stems, stem_info, song_no, frame_no, ex_no, opt)

        # make all combinations of 1 generated stem, with max 6 generated stems, if 4 or more stems
        if len(stem_IDs_in_frame) > 4:
            # at most 8 examples
            generable_stems = random.sample(list(valid_gen_non_bgnd_stem_IDs), np.min([len(valid_gen_non_bgnd_stem_IDs), 8]))
            
            if opt["verbose"]: 
                print(f"\t\tMaking {len(generable_stems)} example(s) out of {len(stem_IDs_in_frame)} stems in frame: ")
            for gen_stem in generable_stems:
                generated_stem = dict({gen_stem: stem_arrs_by_frame[frame_no][gen_stem]})
                source_stems_name = stem_IDs_in_frame[stem_IDs_in_frame != gen_stem]
                source_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in source_stems_name])
                ex_no = make_train_example(source_stems, generated_stem, stem_info, song_no, frame_no, ex_no, opt)

        # make some combinations of 2 generated stems, with max 4 examples, if 5 or more stems
        # TODO: port over narrowing of gen-instruments to this more complex case
        # if len(stems_in_frame) > 5:
        #     # at most 4 examples
        #     generable_stems = random.sample(list(stems_in_frame), np.min([len(stems_in_frame), 5]))
        #     if opt["verbose"]:
        #         print(f"\tMaking {len(generable_stems)-1} examples out of {len(stems_in_frame)} stems in frame: ")
        #     for i in range(len(generable_stems) - 1):
        #         generated_stems = dict({generable_stems[i]: stem_arrs_by_frame[frame_no][generable_stems[i]],
        #                             generable_stems[i+1]: stem_arrs_by_frame[frame_no][generable_stems[i+1]]})
        #         source_stems_name = stems_in_frame[(stems_in_frame != generable_stems[i]) * (stems_in_frame != generable_stems[i+1])]
        #         source_stems = dict([(f, stem_arrs_by_frame[frame_no][f]) for f in source_stems_name])
        #         ex_no = make_train_example(source_stems, generated_stems, stem_info, song_no, frame_no, ex_no, opt)

        if opt["verbose"]: 
            print(f"\n\tFRAME SUMMARY: frame no {frame_no+1} made {ex_no} frames")
        num_examples_total += ex_no

# script information
time_elapsed = (time() - time_start) / 60
print(f"""Preprocessing complete! Summary:
      - preprocessed {len(train_example_dirs)} songs
      - generated {num_examples_total} examples total
      - control methods:", {opt["control_methods"]}
      - runtime: {time_elapsed} min"""
    )