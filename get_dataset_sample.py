from cnet_riff_dataset import CnetRiffDataset
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--promptfile",
    type=str,
    nargs="?",
    default="prompt-sobeldenoise.json",
    help="prompt file where all training info is."
)
parser.add_argument(
    "--data_dir",
    type=str,
    nargs="?",
    default="train-data/",
    help="root directory where all data is"
)
args = parser.parse_args()

# load training dataset and see if contents are as expected
train_dataset = CnetRiffDataset(args.data_dir, promptfile=args.promptfile)
print(f"Dataset loaded. Len of dataset: {len(train_dataset)}")
print("Sample contents of dataset:")
item = train_dataset[0]
plt.imshow((item['jpg'] + 1 )/ 2)
plt.title("Target spectrogram")
plt.figure()
plt.imshow(item['hint'])
plt.title("Source (canny edges)")
print("prompt:", item['txt'])
plt.show()