
# uncomment these lines if github not yet cloned
git clone https://github.com/zachary-shah/mel-train.git
cd mel-train
conda env create -f envs/control_env.yml
conda activate control
python cnet_riff_add_control.py


python cnet_riff_training.py --train_data_dir "train-data/" --max_steps 100000 --save_gpu_memory True
python upload_checkpoints.py 4