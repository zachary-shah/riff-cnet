# main shell script to run train data

# first, make condas environment appropriate for creating all data
conda env create -f envs/processing_env.yml
conda env create -f envs/control_env.yml

# run data processing script in training environment
conda activate processing_env
python make_cnet_dataset.py --audio_dir "raw-audio/" --train_data_dir "train-data/" --prompt_file "prompt_labels.json" --limit 3
conda deactivate

# now run training script in training environment
conda activate control_env
python cnet_riff_training.py --train_data_dir "train-data/"
