import pydub
import numpy as np 
import librosa 
import random

"""
Function for getting the name of the instrument from the metadata
"""
def get_instrument_str(stem_metadata: dict) -> str:
    instrument = stem_metadata["inst_class"]
    midi_program = stem_metadata["midi_program_name"]

    if instrument == "Piano": 
        if midi_program == "Harpsichord":
            instrument = midi_program
        elif midi_program == "Clavinet":
            instrument = midi_program
        elif "Grand Piano" in midi_program:
            instrument = "Grand Piano"
    elif instrument == "Chromatic Percussion":
        instrument = midi_program
    elif instrument == "Organ":
        if "Accordion" in midi_program: 
            instrument = midi_program
        elif midi_program == "Harmonica":
            instrument = midi_program
        else:
            instrument = midi_program
    elif instrument == "Guitar":
        if "Electric" in midi_program:
            instrument = "Electric Guitar"
        elif "Acoustic" in midi_program:
            instrument = "Acoustic Guitar"
        elif "harmonics" in midi_program:
            instrument = midi_program
        elif "Distortion" in midi_program:
            instrument = midi_program
    elif instrument == "Bass":
        if "Acoustic" in midi_program:
            instrument = "Acoustic Bass"
        elif "Electric" in midi_program:
            instrument = "Electric Bass"
        elif "Slap" in midi_program:
            instrument = "Slap Bass"
        elif "Synth" in midi_program:
            instrument = "Synth Bass"
    elif instrument == "Strings":
        if "Strings" in midi_program:
            instrument = "Strings"
        elif "Harp" in midi_program:
            instrument = "Harp"
        else: 
            instrument = midi_program
    elif instrument == "Strings (continued)":
        if "Voice" in midi_program or "Choir" in midi_program:
            instrument = "Synth Voice"
        else:
            instrument = "Strings"
    elif instrument == "Brass":
        if "Synth" in midi_program:
            instrument = "Synth Brass"
        elif "Muted" in midi_program:
            instrument = "Trumpet"
        elif "Section" in midi_program:
            instrument = "Brass instruments"
        else:
            instrument = midi_program
    elif instrument == "Reed":
        instrument = midi_program
    elif instrument == "Pipe":
        instrument = midi_program
    elif instrument == "Synth Lead":
        if "square" in midi_program:
            instrument = "Synth Square Wave Lead"
        elif "sawtooth" in midi_program:
            instrument = "Synth Sawtooth Wave Lead"
        else:
            instrument = "Synth Lead"
    elif instrument == "Ethnic":
        instrument = midi_program
    elif instrument == "Percussive":
        instrument = midi_program
    
    return instrument

"""
Given a segment of audio, return the frames for that stem with sufficient power indexed as a dictionary 
"""
def get_stem_frames(segment: pydub.AudioSegment, 
                    overlap: int = 0,
                    frame_seconds: int = 5.11,
                    min_power_prop: int = 0.5,
                    fs: int = 44100)-> dict:
    
    # get samples and time vec as numpy array
    samples = np.frombuffer(segment.raw_data, dtype=np.int16)

    # get power for samples 
    stem_power = np.mean(np.square(samples[(samples != 0)]))

    # get frames
    num_samples_per_segment = int(frame_seconds * fs)
    hop_length = int(num_samples_per_segment*(1-overlap))
    frames = librosa.util.frame(samples, frame_length=num_samples_per_segment, hop_length=hop_length).T

    # calculate frame powers
    frame_powers = []
    for i in range(frames.shape[0]):
        frame_powers.append(np.mean(np.square(frames[i])))

    # filter frames by ones with low power
    valid_frames = frames[frame_powers > min_power_prop*stem_power, :]
    valid_frame_inds = np.where(frame_powers > min_power_prop*stem_power)[0]

    # return frames as dict indexed by index
    valid_frames = {}
    for ind in valid_frame_inds:
        valid_frames[ind] = frames[ind,:]
    
    return valid_frames

"""
Make a prompt given a list of strings of the generated instruments, and a list of strings of the source instruments
- Will include the list of source instruments as a hint with probability p_src_hint
"""
def make_slakh_prompt(gen_instruments: list,
                      src_instruments: list,
                      p_src_hint: float=0.5) -> str:

    # add source instruments hint p_src_hint*100% of the time
    hint_str = ""
    if np.random.rand() < p_src_hint and len(src_instruments) >= 1: 
        hint_str = "Given "
        if len(src_instruments) == 1:
            hint_str += src_instruments[0] + ": "
        elif len(src_instruments) == 2:
            hint_str = hint_str + src_instruments[0] + " and " + src_instruments[1] + ": "
        else:
            for i in range(len(src_instruments)):
                hint_str += str(src_instruments[i])
                if i == len(src_instruments) - 2:
                    hint_str += ", and "
                elif i == len(src_instruments) - 1:
                    hint_str += ": "
                else:
                    hint_str += ", "

    # get the instruments needed to generate
    if len(gen_instruments) == 1:
        if gen_instruments[0] == "Drums":
            gen_str = random.sample(["Generate a Drum rhythm.",  "Generate Drums."], 1)[0]
        else:
            last_word = random.sample([" stem", " line", " part", ""], 1)[0]
            gen_str = f"Generate a {gen_instruments[0]}{last_word}."
    elif len(gen_instruments) == 2:
        gen_str = "Generate " + gen_instruments[0] + " and " + gen_instruments[1] + "."
    else:
        gen_str = "Generate "
        for (i, instrument) in enumerate(gen_instruments):
            if i == len(gen_instruments) - 1:
                gen_str += f"and {instrument}."
            else:
                gen_str += f"{instrument}, "
    return hint_str + gen_str