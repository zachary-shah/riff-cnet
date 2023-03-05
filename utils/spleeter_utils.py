import logging
import os
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import soundfile as sf
import numpy as np
import librosa

valid_stem_counts = [2, 4, 5]

def verify_stem_count(stem_num):
    if stem_num not in [2, 4, 5]:
        logging.warning(f"Cannot use {stem_num}: valid options are {valid_stem_counts}. Using 2.")
        return False
    return True

def separate_audio_to_dir(inp_audio_path, out_dir, stem_num=2):
    """Given an audio file path, saves the stems of the audio to the specified output folder.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    stem_num = verify_stem_count(stem_num) 
        
    separator = Separator("spleeter:2stems")
    separator.separate_to_file(inp_audio_path, out_dir)
    
def separate_audio(inp_audio_path, fs, stem_num=2):
    """Given an audio file path, returns a dictionary containing numpy arrays associated with different components of the input audio track. Also appends an array with the original full audio to the dictionary, with key "full_audio".
    """
    stem_num = verify_stem_count(stem_num)
    
    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(inp_audio_path, sample_rate=int(fs))
        
    separator = Separator("spleeter:2stems")
    pred_audio_stem = separator.separate(waveform)
    pred_audio_stem["full_audio"] = np.asarray(waveform)
    
    pred_audio_stem["accompaniment"] = librosa.to_mono(pred_audio_stem["accompaniment"].T)
    pred_audio_stem["vocals"] = librosa.to_mono(pred_audio_stem["vocals"].T)
    
    return pred_audio_stem
    
def debug_audio_pred_dict(pred_audio_stem, stem_name, fs):
    if stem_name not in pred_audio_stem.keys():
        logging.error(f"Cannot debug stem of type {stem_name}: valid keys are {pred_audio_stem.keys()}")
        return
    
    desired_stem = pred_audio_stem[stem_name]
    os.makedirs("debug/", exist_ok=True)
    sf.write(f'debug/debug_{stem_name}.wav', desired_stem, samplerate=int(fs))

if __name__ ==  '__main__':
    test_file_path = "../pop-data/pop.00000.wav"
    # test_file_path = "../playground/output_stems/pop.00000/accompaniment.wav"

    fs = 44.1e3
    
    logging.basicConfig(level=logging.INFO)
    
    separate_audio_to_dir(test_file_path, "output_stems/", 2)
    logging.info("Saved audio to directory")
    
    info = separate_audio(test_file_path, fs)
    logging.info("Separated audio to directory")

    print("Vocal Norm:")
    print(np.linalg.norm(info["vocals"]))
    print("Full Audio Norm:")
    print(np.linalg.norm(info["full_audio"]))
    print("Accompaniment Norm:")
    print(np.linalg.norm(info["accompaniment"]))

    
    # logging.info(info["vocals"][:100])
    # logging.info(info["accompaniment"][:100])
    logging.info(np.shape(info["vocals"]))
    logging.info(np.shape(info["accompaniment"]))
    logging.info(np.shape(info["full_audio"]))
    
    debug_audio_pred_dict(info, "vocals", fs)
    debug_audio_pred_dict(info, "full_audio", fs)
    logging.info("Debugged audio vocals, full_audio")