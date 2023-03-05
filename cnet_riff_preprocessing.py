# FUNCTIONS FOR DATA PREPROCESSING

import json
import cv2
import numpy as np
import os
from pathlib import Path

# for spectrum generation
from utils.audio_segment_utils import segment_audio, write_wav_file
from utils import riffusion_utils
from utils import spleeter_utils

# create fresh prompt file
def create_prompt_file(rootdir):
    # remove prompts.json if it already exists; rewrite it fresh
    if Path(os.path.join(rootdir,"prompt.json")).is_file():
        os.remove(os.path.join(rootdir,"prompt.json"))
    return

# append for all segment source/targets created for one audio file
# all sources and files have same prompt for now
def append_to_prompt_file(rootdir, source_filepaths, target_filepaths, prompt, verbose=False):

    with open(os.path.join(rootdir,"prompt.json"), 'a') as outfile:
        for i in range(len(source_filepaths)):
            packet = {
                "source": str(source_filepaths[i]),
                "target": str(target_filepaths[i]),
                "prompt": str(prompt)
            }
            json.dump(packet, outfile)
            outfile.write('\n')
    outfile.close()
    if verbose:
        print(f"Successfully generated prompts for {i} training examples")
    return

def generate_and_replace_canny_source(file_path, low_thres=100, high_thres=200):

    # check threshold values 
    assert low_thres > 1 and low_thres<255, f"Threshold out of bounds; must be between 1 and 255"
    assert low_thres > 1 and high_thres<255, f"Threshold out of bounds; must be between 1 and 255"
    
    # open image
    # print(file_path, type(file_path))
    accompaniment_spec = cv2.imread(str(file_path))
    # flip color scheme
    accompaniment_spec = cv2.cvtColor(accompaniment_spec, cv2.COLOR_BGR2RGB)
    # run canny edge detection
    source_spec = cv2.Canny(accompaniment_spec, low_thres, high_thres)
    cv2.imwrite(str(file_path), source_spec)
    
    return

# given audio files, save all targets, source, and prompt file
def preprocess_batch(audio_files, audio_files_dir, output_dir, prompt_file_path=None, fs=44100, verbose=False, save_wav=False):

    assert prompt_file_path is not None

    #create_prompt_file(rootdir=output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    segments_dir = os.path.join(output_dir,"segment")
    os.makedirs(segments_dir, exist_ok=True)
    targets_dir = os.path.join(output_dir,"target")
    os.makedirs(targets_dir, exist_ok=True)
    sources_dir = os.path.join(output_dir,"source")
    os.makedirs(sources_dir, exist_ok=True)

    # get all prompts in prompt_file as dictionary
    prompt_dict = {}
    p_count = 0
    with open(prompt_file_path, 'r') as prompt_file:
        for line in prompt_file:
            data = json.loads(line)
            prompt_dict[data['file']] = data['prompt']
            p_count += 1
    if verbose: print(f"Read {p_count} files from {prompt_file_path}.")
    prompt_file.close()
    
    for audio_file in audio_files:
        audio_filename = audio_file[:audio_file.index(".wav")]

        # audio splitting
        splits = spleeter_utils.separate_audio(os.path.join(audio_files_dir, audio_file), fs=fs, stem_num=2)
        accompaniment_audio = splits['accompaniment']
        full_audio = splits['full_audio']
        vocal_audio = splits['vocals']

        # get audio segments with pitch augmentation on (should be 72 segments total)
        full_audio_segments = segment_audio(full_audio, fs=fs, num_segments=5, pitch_augment=True)
        accompaniment_audio_segments = segment_audio(accompaniment_audio, fs=fs, num_segments=5, pitch_augment=True)
        vocal_audio_segments = segment_audio(vocal_audio, fs=fs, num_segments=5, pitch_augment=True)

        # remove segments with low vocal power
        acceptable_inds = []
        for i, accompaniment_audio_segment in enumerate(accompaniment_audio_segments):
                if np.linalg.norm(vocal_audio_segments[i]) > np.linalg.norm(accompaniment_audio_segment)*0.1:
                    acceptable_inds.append(i)
                else:
                    if verbose: print("Vocals not detected in segement " + str(i))
        full_audio_segments = full_audio_segments[acceptable_inds]
        accompaniment_audio_segments = accompaniment_audio_segments[acceptable_inds]
        vocal_audio_segments = vocal_audio_segments[acceptable_inds]

        if verbose:
            print(f"Total number of segments for {audio_filename}: {full_audio_segments.shape[0]}")

        # generally, don't save .wav files as this is will require too much storage
        if save_wav:
            full_clip_dir = os.path.join(output_dir,"full_clips")
            os.makedirs(full_clip_dir, exist_ok=True)
            write_wav_file(accompaniment_audio, os.path.join(full_clip_dir, f'{audio_filename}_seg_test_bgnd.wav'), fs=fs,  verbose=verbose)
            write_wav_file(vocal_audio, os.path.join(full_clip_dir, f'{audio_filename}_seg_test_voc.wav'), fs=fs,  verbose=verbose)
            write_wav_file(full_audio, os.path.join(full_clip_dir, f'{audio_filename}_seg_test_full.wav'), fs=fs,  verbose=verbose)

            for i, accompaniment_audio_segment in enumerate(accompaniment_audio_segments):
                write_wav_file(accompaniment_audio_segments[i], os.path.join(segments_dir, f'{audio_filename}_seg{i}_bgnd.wav'), fs=fs,  verbose=verbose)
                write_wav_file(full_audio_segments[i], os.path.join(segments_dir, f'{audio_filename}_seg{i}_full.wav'), fs=fs,  verbose=verbose)

        # make paths for saving targets
        target_save_paths = []
        for i in range(full_audio_segments.shape[0]):
            target_save_paths.append(Path(os.path.join(targets_dir, f'{audio_filename}_seg{i}.jpg')))
    
        # save target spectrograms
        riffusion_utils.audio_to_images_batch(audio_segment_arr = full_audio_segments,
                                             audio_paths = target_save_paths,
                                             sample_rate = fs)
        
        # save source spectrograms
        source_save_paths = []
        for i in range(accompaniment_audio_segments.shape[0]):
            source_save_paths.append(Path(os.path.join(sources_dir, f'{audio_filename}_seg{i}.jpg')))
        riffusion_utils.audio_to_images_batch(audio_segment_arr = accompaniment_audio_segments,
                                             audio_paths = source_save_paths,
                                             sample_rate = fs)
        
        # turn sources into canny edges
        for path in source_save_paths:
            generate_and_replace_canny_source(path, low_thres=100, high_thres=200)
        
        # get prompt for this song
        if audio_file in prompt_dict:
            song_prompt = prompt_dict[audio_file]
        else:
            song_prompt = "Generate a pop melody."
        
        if verbose:
            print(f"  Using prompt \'{song_prompt}\' for file {audio_file}.")

        # append to prompt file for all segments
        append_to_prompt_file(rootdir=output_dir, 
                            source_filepaths=source_save_paths, 
                            target_filepaths=target_save_paths,
                            prompt=song_prompt,
                            verbose=verbose)
    if verbose:
        print("Segmentation and spectrogram generation complete.")
    return

