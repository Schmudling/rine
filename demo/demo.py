import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.utils import get_transforms, get_our_trained_model
from PIL import Image

device = "cuda:0"
_, transforms, _ = get_transforms()
model = get_our_trained_model(ncls=4, device=device)
model.to(device)
model.eval()

real_path = "/content/images/dalle-3/0_0.jpg"
real_image = Image.open(real_path).convert("RGB")
real_tensor = transforms(real_image).unsqueeze(0).to(device)
real_logit = model(real_tensor)[0]
real_probability = torch.sigmoid(real_logit)
print(
    f"real image - prob. to be fake: {real_probability.detach().cpu().numpy()[0][0]*100:1.1f}%"
)

fake_path = "/content/images/dalle-3/0_0.jpg"
fake_image = Image.open(fake_path).convert("RGB")
fake_tensor = transforms(fake_image).unsqueeze(0).to(device)
fake_logit = model(fake_tensor)[0]
fake_probability = torch.sigmoid(fake_logit)
print(
    f"fake image - prob. to be fake: {fake_probability.detach().cpu().numpy()[0][0]*100:1.1f}%"
)

