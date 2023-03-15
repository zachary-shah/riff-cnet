import logging
import os
import sys
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import soundfile as sf
import numpy as np
import librosa
from PIL import Image, ImageOps
from pathlib import Path
import re
import json
import shutil

from utils.spleeter_utils import separate_audio, audio_pred_dict

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_converter import SpectrogramConverter
from riffusion.spectrogram_params import SpectrogramParams

if __name__ ==  '__main__':
    plot_only = True
    if "-g" in sys.argv:
        plot_only = False

    model_audio_folder = "image_log/"
    if not plot_only:
        num_total = 50 # sample 10 images from the image folder
        audio_out_folder = "temp_out_audio2/"
        filenames = os.listdir(model_audio_folder)

        condition_files = [elem for elem in filenames if elem.startswith("conditioning")]
        source_files = [elem for elem in filenames if elem.startswith("source")]
        reconstruction_files = [elem for elem in filenames if elem.startswith("reconstruction")]
        samples_files = [elem for elem in filenames if elem.startswith("samples")]

        img_converter = SpectrogramImageConverter(SpectrogramParams(sample_rate=22050, min_frequency=0, max_frequency=10000))

        os.makedirs(audio_out_folder, exist_ok=True)

        samples_gend = []

        # Generate some samples and such for us to listen to, if we want
        for i in np.linspace(0, len(reconstruction_files)-1, num_total, dtype=int):
            recon_filepath = os.path.join(model_audio_folder, reconstruction_files[i])
            sample_filepath = os.path.join(model_audio_folder, samples_files[i])
            recon_img =  ImageOps.grayscale(Image.open(recon_filepath).convert("RGB")).convert("RGB")
            sample_img = ImageOps.grayscale(Image.open(sample_filepath).convert("RGB")).convert("RGB")
            
            recon_img.save(os.path.join(audio_out_folder, Path(recon_filepath).stem.strip() + "grayed.png"))
            sample_img.save(os.path.join(audio_out_folder, Path(sample_filepath).stem.strip() + "grayed.png"))
            
            out_audio_recon = img_converter.audio_from_spectrogram_image(recon_img, apply_filters=True).set_channels(1)
            out_audio_sample = img_converter.audio_from_spectrogram_image(sample_img, apply_filters=True).set_channels(1)
            
            out_audio_recon.export(os.path.join(audio_out_folder, f"{Path(recon_filepath).stem}.wav"), format="wav")
            out_audio_sample.export(os.path.join(audio_out_folder, f"{Path(sample_filepath).stem}.wav"), format="wav")
            print(f"Completed audio sample {i}", flush=True)
            
            samples_gend.append(recon_filepath)
            samples_gend.append(sample_filepath)
        
    plot_indices = ["014715", "018276"] # Select the gs steps for which we have cool results!
    text_prompts = ["Sample Caption 1", "Sample Caption 2", "Sample Caption 3", "Sample Caption 4", "Sample Caption 5"]

    def get_prefix_fpath(prefix, gs_index_str, search_dir):
        pattern = re.compile(f"{prefix}.*{gs_index_str}")
        sampled_files = os.listdir(search_dir)
        return [os.path.join(search_dir, f) for f in sampled_files if pattern.match(f)][0]

    fig, ax = plt.subplot_mosaic([
        ['tgt_gs'+str(i) for i in range(1, len(plot_indices)+1)],
        ['src_gs'+str(i) for i in range(1, len(plot_indices)+1)],
        ['sample_gs'+str(i) for i in range(1, len(plot_indices)+1)],
        
    ], figsize=(12, 6))
    fig.tight_layout(h_pad = -1.5, w_pad=-19)
    # fig.subplots_adjust(top=0.88)
    # fig.suptitle("Spectrogram Evolution: Target to Edges to Model", fontsize=30)

    for i, idx in enumerate(plot_indices):
        i = i+1
        tgt_fpath = get_prefix_fpath("reconstruction", idx, model_audio_folder)
        sample_fpath = get_prefix_fpath("samples", idx, model_audio_folder)
        prompt_fpath = get_prefix_fpath("conditioning", idx, model_audio_folder)
        
        # Find the source image for this global step using JSON
        corrupted_src_fpath = get_prefix_fpath("control", idx, model_audio_folder)
        f = open("image_log/_corruption_match_table.json")
        img_dict = json.load(f)
        src_fpath = img_dict[corrupted_src_fpath.split("/")[1]]
        f.close()
        
        # Plot the images
        src_img = plt.imread(src_fpath)
        tgt_img = plt.imread(tgt_fpath)
        sample_img = plt.imread(sample_fpath)
        
        ax[f'tgt_gs{i}'].imshow(tgt_img)
        ax[f'tgt_gs{i}'].axis('off')
        
        ax[f'src_gs{i}'].imshow(src_img, cmap='gray')
        ax[f'src_gs{i}'].axis('off')

        ax[f'sample_gs{i}'].imshow(sample_img)
        ax[f'sample_gs{i}'].axis('off')
        
    # All annotations relative to a single subfigure
    ax[f'tgt_gs2'].annotate('Target Spectrograms', xy=(90, 480), xycoords='figure pixels', wrap=True, ha='center')
    ax[f'tgt_gs2'].annotate('Generated\nBackground Edges', xy=(90, 300), xycoords='figure pixels', wrap=True, ha='center')
    ax[f'tgt_gs2'].annotate('Model Output\nSpectrogram', xy=(90, 120), xycoords='figure pixels', wrap=True, ha='center')

    height = 10
    incr = 182
    base = 70
    ax[f'tgt_gs2'].annotate('Model Train\nPercentage:', xy=(90, height), xycoords='figure pixels', wrap=True, ha='center')
    ax[f'tgt_gs2'].annotate('~10%', xy=(base+incr, height), xycoords='figure pixels', wrap=True, ha='center')
    ax[f'tgt_gs2'].annotate('~30%', xy=(base+2*incr, height), xycoords='figure pixels', wrap=True, ha='center')
    ax[f'tgt_gs2'].annotate('~50%', xy=(base+3*incr, height), xycoords='figure pixels', wrap=True, ha='center')
    ax[f'tgt_gs2'].annotate('~70%', xy=(base+4*incr, height), xycoords='figure pixels', wrap=True, ha='center')
    ax[f'tgt_gs2'].annotate('~100%', xy=(base+5*incr, height), xycoords='figure pixels', wrap=True, ha='center')

    plt.savefig("spec_evolution.png")

    if "-a" in sys.argv:
        # Collect audio samples for each of the final chosen images
        audio_folder_name = "final_audio/"
        # if os.path.isdir(audio_folder_name):
        #     shutil.rmtree(audio_folder_name)
            
        img_converter = SpectrogramImageConverter(SpectrogramParams(sample_rate=41000, min_frequency=0, max_frequency=10000))

        os.makedirs(audio_folder_name, exist_ok=True)
        for i, idx in enumerate(plot_indices):
            tgt_fpath = get_prefix_fpath("reconstruction", idx, model_audio_folder)
            sample_fpath = get_prefix_fpath("samples", idx, model_audio_folder)
            prompt_fpath = get_prefix_fpath("conditioning", idx, model_audio_folder)
            
            # Find the source image for this global step using JSON
            corrupted_src_fpath = get_prefix_fpath("control", idx, model_audio_folder)
            f = open("image_log/_corruption_match_table.json")
            img_dict = json.load(f)
            src_fpath = img_dict[corrupted_src_fpath.split("/")[1]]
            f.close()
            
            for elem in [tgt_fpath, sample_fpath]:
                img_object = ImageOps.grayscale(Image.open(elem).convert("RGB")).convert("RGB")
                audio = img_converter.audio_from_spectrogram_image(img_object, apply_filters=True).set_channels(1)
                curr_out_audio_path = f"{Path(elem).stem}.wav"
                audio.export(os.path.join(audio_folder_name, curr_out_audio_path), format="wav")
                
                fs = 44100
                info = separate_audio(os.path.join(audio_folder_name, curr_out_audio_path), fs=False)
                audio_pred_dict(info, "vocals", os.path.join(audio_folder_name, f"{Path(elem).stem}_vocals.wav"), fs)
                audio_pred_dict(info, "accompaniment", os.path.join(audio_folder_name, f"{Path(elem).stem}_bgnd.wav"), fs)
                audio_pred_dict(info, "full_audio", os.path.join(audio_folder_name, f"{Path(elem).stem}_full.wav"), fs)
                logging.info(f"Debugged audio vocals, full_audio for {i}", flush=True)
                