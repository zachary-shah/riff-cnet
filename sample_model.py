import os, zipfile, glob, argparse
import numpy as np
import cv2
from PIL import Image

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

from cldm.ddim_hacked import DDIMSampler
from cnet_riff_dataset import CnetRiffDataset

from utils.cnet_utils import get_model, sample_ddim


parser = argparse.ArgumentParser()
parser.add_argument(
    "--control_methods",
    type=str,
    nargs="+",
    default=["fullspec", "canny", "sobel", "sobeldenoise"],
    help="control method(s) to develop for preprocessing. input like --control_methods \"method1\" \"method2\" ... \"methodn\"."
)
parser.add_argument(
    "--model_paths",
    type=str,
    nargs="+",
    default=["models/cnet-riff-final.ckpt"],
    help="model paths for each control method(s). input like --model_paths \"path1\" \"path2\" ... \"pathn\"."
)
parser.add_argument(
    "--root_save_dir",
    type=str,
    nargs="?",
    default='model_samples',
    help="directory name to save all samples in"
)
parser.add_argument(
    "--val_dataset_path",
    type=str,
    nargs="?",
    default='val-data',
    help="where to save data to"
)
parser.add_argument(
    "--num_samples",
    type=int, 
    nargs="?",
    default=2,
    help="number of samples to generate per val example"
) 
parser.add_argument(
    "--max_examples",
    type=int, 
    nargs="?",
    default=15,
    help="max number of samples to generate data for"
) 
parser.add_argument(
    "--skip_factor",
    type=int,
    nargs="?",
    default=1,
    help="skip over val samples in sample iterations."
) 
parser.add_argument(
    "--zip_data",
    type=bool,
    nargs="?",
    default=False,
    help="True to create .zip file with all data at the end of script, with same name as root_save_dir"
) 
parser.add_argument(
    "--sample_bgnd",
    type=bool,
    nargs="?",
    default=False,
    help="True to get samples of background audio for each example. This requires val_dataset to have subfolder titled \"bgnd\" which has all the sample backgrounds saved to it."
) 

opt = parser.parse_args()

assert not os.path.exists(opt.root_save_dir)
assert len(opt.model_paths) == len(opt.control_methods)

img_converter_to_audio = SpectrogramImageConverter(SpectrogramParams(sample_rate=44100, min_frequency=0, max_frequency=10000))

# generate the samples for the desired dataset
os.makedirs(opt.root_save_dir, exist_ok=False)

for k, control_method in enumerate(opt.control_methods):

    # get new model
    model = get_model(opt.model_paths[k])
    ddim_sampler = DDIMSampler(model)
    save_dir = os.path.join(opt.root_save_dir,control_method)
    os.makedirs(save_dir, exist_ok=True)

    # get dataset
    val_dataset = CnetRiffDataset(opt.val_dataset_path, promptfile="prompt-"+control_method+".json")

    print(f"{control_method} model loaded!")

    for i, item in enumerate(val_dataset):
        if i % opt.skip_factor == 0:
            # only sample a subset, like around 15 or so samples should be good
            print(f"Sampling for prompt: {item['txt']}")
            results, _ = sample_ddim(item['hint'], 
                                    item['txt'], 
                                    model, 
                                    ddim_sampler, 
                                    num_samples=opt.num_samples, 
                                    control_lims=[0.0,1.0])

            for (k, sample) in enumerate(results):
                # save each sample spectrogram
                cv2.imwrite(os.path.join(save_dir,f"{item['txt']}_samp_{k}.png"), sample)
                # save each sample audio
                sample_img = Image.open(os.path.join(save_dir,f"{item['txt']}_samp_{k}.png"))
                out_audio_recon = img_converter_to_audio.audio_from_spectrogram_image(sample_img, apply_filters=True).set_channels(2)
                out_audio_recon.export(os.path.join(save_dir,f"{item['txt']}_samp_{k}.wav"), format="wav")

            # save source for reference
            source = item['hint']
            if (np.max(source) <= 1) and (np.min(source) >= 0):
                print("switching control scale from [0.,1.] to [0,255]")
                source = np.uint8(source  * 255)
            cv2.imwrite(os.path.join(save_dir,f"{item['txt']}_source.png"), source)

            if opt.sample_bgnd:
                # save bgnd image and audio if possible
                bgnd_path = item['name'].replace("target", "bgnd")
                bgnd = cv2.imread(bgnd_path)
                cv2.imwrite(os.path.join(save_dir,f"{item['txt']}_bgnd.png"), bgnd)
                bgnd_img = Image.open(os.path.join(save_dir,f"{item['txt']}_bgnd.png")) 
                out_audio_recon = img_converter_to_audio.audio_from_spectrogram_image(bgnd_img, apply_filters=True).set_channels(2)
                out_audio_recon.export(os.path.join(save_dir,f"{item['txt']}_bgnd.wav"), format="wav") 

            # save target too
            target = (item['jpg'] + 1) / 2 * 255
            cv2.imwrite(os.path.join(save_dir,f"{item['txt']}_target.png"), target)
            target_img = Image.open(os.path.join(save_dir,f"{item['txt']}_target.png")) 
            out_audio_recon = img_converter_to_audio.audio_from_spectrogram_image(target_img, apply_filters=True).set_channels(2)
            out_audio_recon.export(os.path.join(save_dir,f"{item['txt']}_target.wav"), format="wav") 

            if i >= opt.max_examples - 1:
                break

    del model
    del ddim_sampler
    del val_dataset

if opt.zip_data:
    zip_path = opt.root_save_dir + '.zip'
    assert not os.path.exists(zip_path)
    print(f"Zipping data to {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w') as f:
        for file in glob.glob(opt.root_save_dir + '/*'):
            f.write(file)

print("Done! Ready for upload.")