
from cnet_riff_dataset import CnetRiffDataset
import matplotlib.pyplot as plt

# load training dataset and see if contents are as expected
train_dataset = CnetRiffDataset("train-data/")
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