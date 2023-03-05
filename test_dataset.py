
from cnet_riff_dataset import CnetRiffDataset

train_dataset = CnetRiffDataset("train-data/")

# # show sample contents if desired
print("Sample contents of dataset")
print(f"Len of dataset: {len(train_dataset)}")
# item = train_dataset[0]
# plt.imshow((item['jpg'] + 1 )/ 2)
# plt.title("Target spectrogram")
# plt.figure()
# plt.imshow(item['hint'])
# plt.title("Source (canny edges)")
# plt.show()
# print("prompt:", item['txt'])