import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoImageProcessor, AutoModelForImageClassification

response = requests.get(
    'https://github.com/laxmimerit/dog-cat-full-dataset/blob/master/data/train/cats/cat.10055.jpg?raw=true')
img = Image.open(BytesIO(response.content))

img_proc = AutoImageProcessor.from_pretrained(
    'google/vit-base-patch16-224')
model = AutoModelForImageClassification.from_pretrained(
    'google/vit-base-patch16-224')

inputs = img_proc(img, return_tensors='pt')

with torch.no_grad():
    logits = model(**inputs).logits

predicted_id = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_id]
print(predicted_id, '-', predicted_label)

