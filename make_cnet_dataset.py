import argparse, os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
from cnet_riff_dataset import CnetRiffDataset, preprocess_batch

# how to run:
# python make_cnet_dataset.py --audio_dir "../pop-data/" --train_data_dir "train-data/" --prompt_file "prompt_labels.json" --limit 2

def main():
    # names of files to segment
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_dir",
        type=str,
        nargs="?",
        default="raw-audio/",
        help="directory where all training .wav files located. will use every audio file in directory for training."
    )

    parser.add_argument(
        "--train_data_dir",
        type=str,
        nargs="?",
        default="train-data/",
        help="directory to output training dataset to"
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        nargs="?",
        default="prompt_labels.json",
        help="filepath to json file with all prompts for training data"
    )

    parser.add_argument(
        "--limit",
        type=int, 
        nargs="?",
        default="-1",
        help="limit of number of files to process. -1 for no limit"
    ) 

    parser.add_argument(
        "--show_sample",
        type=bool,
        nargs="?",
        default="False",
        help="True to show a sample from the dataloader"
    ) 
    
    args = parser.parse_args()
    audio_files_dir = args.audio_dir
    train_data_dir = args.train_data_dir
    prompt_file = args.prompt_file
    limit = args.limit
    show_sample = args.show_sample
    audio_files = os.listdir(audio_files_dir)

    # generate source and target specgrams
    preprocess_batch(audio_files = audio_files[0:limit],
                    audio_files_dir = audio_files_dir,
                    output_dir = train_data_dir,
                    prompt_file_path = prompt_file,
                    fs=44100,
                    verbose=True,   
                    save_wav=False)

    # collect all training data into training object
    print("Preprocesing complete!")

    if show_sample:
        train_dataset = CnetRiffDataset(train_data_dir)

        # show sample contents if desired
        print("Sample contents of dataset")
        item = train_dataset[0]
        plt.imshow((item['jpg'] + 1 )/ 2)
        plt.title("Target spectrogram")
        plt.figure()
        plt.imshow(item['hint'])
        plt.title("Source (canny edges)")
        plt.show()
        print("prompt:", item['txt'])

if __name__ ==  '__main__':
    main()