git clone https://github.com/zachary-shah/mel-train.git
cd mel-train
conda env create -f envs/control_env.yml
conda activate control
python cnet_riff_add_control.py
python cnet_riff_training.py --train_data_dir "train-data/" --batch_size 4 --accumulate_gradient_batches 1