import torch.nn as nn
import torch
from mtcnn import MTCNN
import torchvision as tv
from agre_utils import postprocess_face, check_image_and_box
import torchvision.transforms.functional as F
from mobnetv1 import MobileNetV1

class AGRE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mtcnn = MTCNN()
        self.mobnet = MobileNetV1()
        self.image_processing = nn.Sequential(
            tv.transforms.Resize((224, 224)),
        )

    def forward(self, x):
        # x = torch.from_numpy(np.array(full_image)).to(device)
        # full_image = Image.fromarray(x.numpy())
        y, _ = self.mtcnn(x)
        boxes, probs = postprocess_face(y)
        res = check_image_and_box(x, boxes, probs)
        if res != 0:
            return {
                'code': res,
                'age': -1,
                'gender': -1,
                'race': -1,
            }
        box = boxes[-1]
        left, top, right, bottom = [i.item() for i in box.int()]
        face = x[top:bottom, left:right, :].permute(2, 0, 1)
        
        face = self.image_processing(face)
        face = face.float()
        face[0, ...] -= 123.68  # R
        face[1, ...] -= 116.779 # G
        face[2, ...] -= 103.939 # B        
        face = face.unsqueeze(0)
    
        age, gender, race = self.mobnet(face)

        return self.process_dict(0, age, gender, race)

    def process_dict(
                self, code: int, age: torch.Tensor, 
                gender: torch.Tensor, race: torch.Tensor
            ):

        # if code != 0:
        #     return {
        #         'code': code,
        #         'age': -1,
        #         'gender': -1,
        #         'race': -1,
        #     }

        return {
            'code': code,
            'age': self.get_age(age),
            'gender': self.get_gender(gender),
            'race': self.get_race(race),
        }
    
    def get_age(self, age_score: torch.Tensor) -> int:
        age_score = age_score.squeeze()
        age_score_sorted, idx_sorted = torch.sort(age_score, descending=True)
        # idx_sorted = age_score.argsort()[::-1]
        # age_score_sorted = age_score[idx_sorted]
        return int(((idx_sorted + 0.5) * age_score_sorted).sum().item())

    def get_gender(self, gender_score: torch.Tensor) -> int:
        gender_score = gender_score.squeeze().item()
        return int(gender_score >= 0.6)

    def get_race(self, race_score: torch.Tensor) -> int:
        race_score = race_score.squeeze()
        return int(race_score.argmax().item())



