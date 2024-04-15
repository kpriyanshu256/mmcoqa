import torchvision
from PIL import Image
import os
import joblib as jb
import json
import torch
from tqdm.auto import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", max_new_tokens=20)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

class Dataset:
    def __init__(self, I):
        self.I = I

    def __len__(self):
        return len(self.I)

    def __getitem__(self, i):
        return self.image_transform(self.I[i])

    def image_transform(self, path):
        img = np.array(Image.open(path).convert("RGB"))
        return img, path

def fn(line):
    try:
        line = json.loads(line.strip())
        path = os.path.join("MMCoQA/final_dataset_images", str(line['path']))
        _ = Image.open(path).convert("RGB")
        return path
    except:
        return ""
 
 
with open("MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl", 'r') as f:
    lines = f.readlines()
    I = jb.Parallel(-1)(jb.delayed(fn)(x) for x in tqdm(lines))
 
I = [x for x in I if x!=""]
res = {}
print(len(I))
ds = Dataset(I)

dl = torch.utils.data.DataLoader(
        ds,
        batch_size= 1,
        num_workers=4,
)


for i, b in tqdm(enumerate(dl), total=len(I)):
    try:
        img, path = b
        inputs = processor(img.squeeze(0), return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        im_name = path[0].split('/')[-1].split('.')[0]
        res[im_name] = processor.decode(out[0], skip_special_tokens=True)
    except:
        print(i)

with open("captions.json", "w+") as f:
    res = json.dumps(res, indent=4)
    f.write(res)