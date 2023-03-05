import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from pydub.playback import play
from spectrogram_image_converter import SpectrogramImageConverter
from spectrogram_params import SpectrogramParams
from PIL import Image
import argparse




def generate_seed(wav_path, save_path, start_time):
    audio = AudioSegment.from_wav(wav_path)
    audio = audio[start_time*1000:start_time*1000+5000]

    if audio.channels > 1:
        mono_audios = audio.split_to_mono()
        audio = (mono_audios[0] + mono_audios[1])/2

    params = SpectrogramParams(False)
    converter  = SpectrogramImageConverter(params)
    image = converter.spectrogram_image_from_audio(segment=audio)
    arr = np.array(image)
    image.save(save_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate Seed Images from wav file')
    parser.add_argument('wavdir', help='Path to the wave file')
    parser.add_argument('--save_path', default= "../new_seed_images/newseed.png", help= 'Path to the image file')
    parser.add_argument('--start_time', type=float, default="0", help='Start time (s) for generating the file')
    args = parser.parse_args()

    wav_path = args.wavdir
    save_path = args.save_path
    start_time = args.start_time
    generate_seed(wav_path, save_path, start_time)
    