import os, argparse, json
import pydub
import numpy as np
from time import time
import torch 
from librosa.effects import pitch_shift

from utils.riffusion_utils import audio_array_to_image
from cnet_riff_preprocessing import append_to_prompt_file, generate_and_save_control
from utils.slakh_utils import get_stem_frames
from utils.audio_segment_utils import write_wav_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "--control_methods",
    type=str,
    nargs="+",
    default=["fullspec", "canny", "sobel", "sobeldenoise"],
    help="control method(s) to develop for preprocessing. input like --control_methods \"method1\" \"method2\" ... \"methodn\"."
)
parser.add_argument(
    "--gen_stem",
    type=str,
    nargs="?",
    default="vocals",
    help="stem to generate. must be one of [vocals, bass, drums, other]."
)
parser.add_argument(
    "--root_data_dir",
    type=str,
    nargs="?",
    default='../musdb18-wav/train',
    help="directory path to slakh dataset to preprocess"
)
parser.add_argument(
    "--data_root",
    type=str,
    nargs="?",
    default='../musdb-preprocessed/train',
    help="where to save data to"
)
parser.add_argument(
    "--prompt_filepath",
    type=str,
    nargs="?",
    default='../musdb18-wav/prompt_labels.json',
    help="json file with prompts for all the songs"
)
parser.add_argument(
    "--verbose",
    type=bool, 
    nargs="?",
    default=True,
    help="true to display preprocessing results in detail"
) 
parser.add_argument(
    "--fs",
    type=int, 
    nargs="?",
    default=44100,
    help="sampling rate"
) 
parser.add_argument(
    "--canny_low_thresh",
    type=int,
    nargs="?",
    default=150,
    help="param for canny edge detection, if using"
) 
parser.add_argument(
    "--canny_high_thresh",
    type=int,
    nargs="?",
    default=200,
    help="param for canny edge detection, if using"
) 
parser.add_argument(
    "--denoise_h",
    type=int,
    nargs="?",
    default=15,
    help="param for sobel nl means denoising, if using"
) 
parser.add_argument(
    "--frame_overlap",
    type=float,
    nargs="?",
    default=0.,
    help="overlap in generated frames, in percent."
) 
parser.add_argument(
    "--frame_len_seconds",
    type=float,
    nargs="?",
    default=5.11,
    help="frame length, in seconds"
) 
parser.add_argument(
    "--frame_min_power_prop",
    type=float,
    nargs="?",
    default=0.4,
    help="minimum power of stem in a frame for power threshold filtering"
) 
parser.add_argument(
    "--device",
    type=str,
    nargs="?",
    default="cpu",
    help="device: either cuda or cpu"
)
parser.add_argument(
    "--save_wav",
    type=bool,
    nargs="?",
    default=False,
    help="true to save .wav files for all examples for debugging"
)
parser.add_argument(
    "--max_examples",
    type=int,
    nargs="?",
    default=np.infty,
    help="Max number of examples to create in total"
)
parser.add_argument(
    "--pitch_augment",
    type=bool,
    nargs="?",
    default=False,
    help="True to augment pitch of each example"
)

opt = parser.parse_args()


def make_train_example(gen_wav, bgnd_wav, prompt, song_no, frame_no, ex_no, opt):

    # make target as combo of source and generated stems
    targ_wav = gen_wav + bgnd_wav

    # path naming 
    example_name = f"s{song_no}_f{frame_no}_e{ex_no}.jpg"
    target_path = os.path.join(opt.data_root, "target", example_name)
    bgnd_path = os.path.join(opt.data_root, "bgnd", example_name)

    # mix target stems and generate/save spectrogram
    target_spectrogram = audio_array_to_image(targ_wav, 
                                  save_img=True,
                                  outpath=target_path[:-4],
                                  sample_rate=opt.fs,
                                  device=opt.device,
                                  image_extension="jpg")
    
    # mix source stems and make spectrogram
    source_spectrogram = audio_array_to_image(bgnd_wav, 
                                  save_img=True,
                                  outpath=bgnd_path[:-4],
                                  sample_rate=opt.fs,
                                  device=opt.device)
    
    # add control
    for control_method in opt.control_methods:
        source_path = os.path.join(opt.data_root, "source-"+control_method, example_name)

        # generate control example for each method desired
        generate_and_save_control(source_spectrogram, source_path, control_method, opt)
        
        # add source example to respective prompt file
        append_to_prompt_file(opt.data_root, [source_path], [target_path], prompt, prompt_filename=f"prompt-{control_method}.json", verbose=False)

    # optionally save audio for debugging
    if opt.save_wav:
        write_wav_file(gen_wav, os.path.join(opt.data_root, 'wav', example_name[:-4]+"_gen"+".wav"), fs=opt.fs)
        write_wav_file(targ_wav, os.path.join(opt.data_root, 'wav', example_name[:-4]+"_targ"+".wav"), fs=opt.fs)
        write_wav_file(bgnd_wav, os.path.join(opt.data_root, 'wav', example_name[:-4]+"_bgnd"+".wav"), fs=opt.fs)

    ex_no += 1
    return ex_no

def get_audio_seg(filepath):
    seg = pydub.AudioSegment.from_file(filepath)
    if seg.channels != 1:
        seg = seg.set_channels(1)
    if seg.frame_rate != opt.fs:
        seg = seg.set_frame_rate(opt.fs)
    assert seg.channels == 1
    assert seg.frame_rate == opt.fs
    return seg
         

# tracking
num_examples_total = 0
time_start = time()

# cuda if possible
if torch.cuda.is_available():
    opt.device = "cuda" 

if opt.verbose:
    print("Beginning preprocessing!")

## INPUT VALIDATION 
STEMS = np.array(["vocals", "bass", "drums", "other"])
assert opt.gen_stem in STEMS, f"Input stem must be one of {STEMS}"
bgnd_stems = STEMS[np.array([opt.gen_stem != stem for stem in STEMS])]

# check for prompt path
assert os.path.exists(opt.prompt_filepath), "Prompt filepath invalid."

# get all prompts in prompt_file as dictionary
prompt_dict = {}
p_count = 0
with open(opt.prompt_filepath, 'r') as prompt_file:
    for line in prompt_file:
        data = json.loads(line)
        prompt_dict[data['file']] = data['prompt']
        p_count += 1
if opt.verbose: print(f"Read {p_count} prompts from prompt_file.json.")

# make all directories needed
os.makedirs(opt.data_root, exist_ok=True)
control_roots = [os.path.join(opt.data_root, "source-"+mthd) for mthd in opt.control_methods]
for control_root in control_roots:
    os.makedirs(control_root, exist_ok=True)
os.makedirs(os.path.join(opt.data_root, 'target'), exist_ok=True)
os.makedirs(os.path.join(opt.data_root, 'bgnd'), exist_ok=True)
if opt.save_wav: os.makedirs(os.path.join(opt.data_root, 'wav'), exist_ok=True)

# data folder
example_dirs = sorted([f for f in os.listdir(opt.root_data_dir) if f != '.DS_Store'])

# TODO: just for prototyping; remove later
example_dirs = example_dirs[:2]

for song_no, example_dir in enumerate(example_dirs):

    if opt.verbose: print(f"Processing song {song_no+1}/{len(example_dirs)}: {example_dir}...")

    # get song prompt
    try:
        prompt = prompt_dict[example_dir]
    except:
        print(f" -- Warning: could not find prompt for {example_dir}. Using default prompt instead.")
        prompt = f"Generate a {opt.gen_stem} stem."

    if opt.verbose: print(f"  Prompt for song: \"{prompt}\"")

    # get audios for given file. make sure they are 1 channel audio
    gen_seg = get_audio_seg(os.path.join(opt.root_data_dir, example_dir, opt.gen_stem+".wav"))
    
    bgnd_seg = get_audio_seg(os.path.join(opt.root_data_dir, example_dir, bgnd_stems[0]+".wav"))
    for i in range(1, len(bgnd_stems)):
        bgnd_seg = bgnd_seg.overlay(get_audio_seg(os.path.join(opt.root_data_dir, example_dir, bgnd_stems[i]+".wav")))

    # make audio frames
    gen_frames = get_stem_frames(gen_seg, 
                            overlap = opt.frame_overlap,
                            frame_seconds = opt.frame_len_seconds,
                            min_power_prop = opt.frame_min_power_prop,
                            fs = opt.fs)

    bgnd_frames = get_stem_frames(bgnd_seg, 
                            overlap = opt.frame_overlap,
                            frame_seconds = opt.frame_len_seconds,
                            min_power_prop = -1,
                            fs = opt.fs)

    valid_frames = np.array(list(gen_frames.keys()))

    if opt.verbose:
        print(f"  FRAMING: ")
        print(f"    Number of valid frames in {opt.gen_stem}: {len(gen_frames)}")
        print(f"    Number of total frames in bgnd: {len(bgnd_frames)}")

    for fno, f in enumerate(valid_frames):
        if opt.verbose: print(f"   Frame {fno+1}/{len(valid_frames)}")
        try:
            ex_no = 0

            ex_no = make_train_example(gen_frames[f], bgnd_frames[f], prompt, song_no, fno, ex_no, opt)

            # modulate through 12 keys
            if opt.pitch_augment:
                if opt.verbose: print("    -- Pitch agumenting frame!")
                # pitch modulation for data augmentation
                for pitch_offset in [-3, -2, -1, 1, 2, 3]:
                    ex_no = make_train_example(pitch_shift(np.squeeze(gen_frames[f].astype('float')), sr=opt.fs, n_steps=pitch_offset),
                                            pitch_shift(np.squeeze(bgnd_frames[f].astype('float')), sr=opt.fs, n_steps=pitch_offset),
                                            prompt,
                                            song_no, 
                                            fno, 
                                            ex_no, 
                                            opt)

            num_examples_total += ex_no
            if num_examples_total >= opt.max_examples: break
        except:
            print(f"      -- WARNING: error making example for frame {f} -- ")

    if num_examples_total >= opt.max_examples:
        print(f"Max example count reached, terminating processing.") 
        break

# script information
time_elapsed = (time() - time_start) / 60
print(f"""Preprocessing complete! Summary:
      - preprocessed {song_no+1} songs
      - generated {num_examples_total} examples total
      - control methods:", {opt.control_methods}
      - runtime: {time_elapsed} min""")