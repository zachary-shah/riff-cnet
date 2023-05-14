import json, os, shutil
import numpy as np 
import random
import pydub
import librosa 
from tqdm import tqdm

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

# given some set of data directories, combine them and split in to train / val sets
def split_train_val(all_data_dirs = ["slakh-preproccessed"], 
                    control_methods = ["canny", "fullspec", "sobel", "sobeldenoise"],
                    train_data_dir = "slakh-data/train/",
                    val_data_dir = "slakh-data/val/",
                    ):
        
    assert not os.path.exists(train_data_dir), "train data already exists."
    assert not os.path.exists(val_data_dir), "val data already exists."

    print("Splitting into train / val datasets: ")

    os.makedirs(train_data_dir)
    os.makedirs(val_data_dir)
    for control_method in control_methods:
        os.makedirs(os.path.join(val_data_dir, "source-" + control_method), exist_ok=True)
        os.makedirs(os.path.join(train_data_dir,"source-" + control_method), exist_ok=True)
    os.makedirs(os.path.join(val_data_dir, "target/"), exist_ok=True)
    os.makedirs(os.path.join(train_data_dir, "target/"), exist_ok=True)

    all_data = {}
    for control_method in control_methods:
        all_data[control_method] = []
        for rootdir in all_data_dirs:
            with open(os.path.join(rootdir, 'prompt-'+control_method+'.json'), 'rt') as f:
                for line in f:
                    all_data[control_method].append(json.loads(line))

    # get random train / test split
    n_total = len(all_data[control_methods[0]])
    n_test = int(n_total * 0.01)
    inds = list(np.arange(n_total))

    test_inds = random.sample(inds, n_test)
    test_inds = np.sort(test_inds)
    train_inds = np.array(inds)[[inds[i] not in test_inds for i in range(n_total)]]

    # move data over for each control method
    for c_num, control_method in enumerate(control_methods):

        train_data = np.array(all_data[control_method])[list(train_inds)]
        test_data = np.array(all_data[control_method])[list(test_inds)]

        print(f"moving for control method: {control_method}")
        print(f"  {len(train_data)} train examples")
        print(f"  {len(test_data)} test examples")

        # copy over train data
        for i in tqdm(range(len(train_data))):
            # only move target for first control method
            if c_num == 0:
                new_target_file = os.path.join(train_data_dir, "target", train_data[i]["target"].split('/')[-1])
                shutil.move(train_data[i]["target"], new_target_file)
            # copy control method
            new_file = os.path.join(train_data_dir, "source-"+control_method, train_data[i]["source"].split('/')[-1])
            shutil.move(train_data[i]["source"], new_file)

        # write train prompt
        with open(os.path.join(train_data_dir,"prompt-"+control_method+".json"), 'w') as outfile:
            for i in range(len(train_data)):
                packet = {
                    "source": str(os.path.join(train_data_dir, "source-"+control_method, train_data[i]["source"].split('/')[-1])),
                    "target":  str(os.path.join(train_data_dir, "target", train_data[i]["target"].split('/')[-1])),
                    "prompt": str(train_data[i]["prompt"])
                }
                json.dump(packet, outfile)
                outfile.write('\n')
            outfile.close()

        # copy over val data
        for i in tqdm(range(len(test_data))):
            # only move target for first control method
            if c_num == 0:
                new_target_file = os.path.join(val_data_dir, "target", test_data[i]["target"].split('/')[-1])
                shutil.move(test_data[i]["target"], new_target_file)
            # copy control method
            new_file = os.path.join(val_data_dir, "source-"+control_method, test_data[i]["source"].split('/')[-1])
            shutil.move(test_data[i]["source"], new_file)

        # write val prompt
        with open(os.path.join(val_data_dir,"prompt-"+control_method+".json"), 'w') as outfile:
            for i in range(len(test_data)):
                packet = {
                    "source": str(os.path.join(val_data_dir, "source-"+control_method, test_data[i]["source"].split('/')[-1])),
                    "target":  str(os.path.join(val_data_dir, "target", test_data[i]["target"].split('/')[-1])),
                    "prompt": str(test_data[i]["prompt"])
                }
                json.dump(packet, outfile)
                outfile.write('\n')
            outfile.close()
    