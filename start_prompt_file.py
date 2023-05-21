import os, json

files = sorted(os.listdir("../musdb18/train") + os.listdir("../musdb18/test"))

with open(os.path.join("../musdb18","prompt_labels.json"), 'a') as outfile:
    for i in range(len(files)):
        packet = {
            "file": str(files[i]),
            "prompt": "Generate a vocal melody.",
        }
        json.dump(packet, outfile)
        outfile.write('\n')
outfile.close()