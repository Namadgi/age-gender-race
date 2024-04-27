import torch
from agre import AGRE
# from PIL import Image
import torchvision
# from torchvision.transforms import functional as F
# import numpy as np
from torch.utils.mobile_optimizer import optimize_for_mobile
# import timm

def load_model():
    model = AGRE()
    model.mtcnn.load_state_dict(torch.load('model_weights/mtcnn.pth', map_location='cpu'))
    model.mobnet.load_state_dict(torch.load('model_weights/mobnet.pth', map_location='cpu'))
    model.eval()
    return model

def save_model(model, x):
    scripted_module = torch.jit.script(model, x)
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("model_weights/agre.ptl")

# efnet = timm.create_model('efficientnet_b0')
# efnet.load_state_dict(torch.load('model_weights/enet_b0_sd.pth', map_location='cpu'))

# model = load_model()
x = torchvision.io.read_image('data/98_b.jpg').permute(1, 2, 0)
# save_model(model, x)
model = torch.jit.load('model_weights/agre.ptl')

with torch.no_grad():
    y = model(x)
print(y)