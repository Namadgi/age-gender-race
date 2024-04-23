import io
import os
import cv2
import json
import base64
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps

import torch
from ts.torch_handler.base_handler import BaseHandler

class AGREHandler(BaseHandler):
    onnx_model = None
    INPUT_SIZE = (224, 224)
    race_labels = ["White", "Black", "Asian", "Indian", "Latino / Middle Eastern"]

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        if data is None:
            return data

        for row in data:
            data = row.get('data') or row.get('body')
        
        if isinstance(data, dict):
            data: dict = data['instances'][0]
            # Download file
            if 'b64' not in data.keys():
                token = data['token']
                bucket_name = data['bucket_name']
                object_name = data['object_name']
                os.system(
                    f'curl -X GET ' +
                    f'-H "Authorization: Bearer {token}" -o {object_name} '
                    f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_name}?alt=media"'
                )

                cap = cv2.VideoCapture(object_name)
                n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                t_frame = n_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, t_frame)
                res = False
                while not res:
                    res, frame = cap.read()
                cap.release()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)
        
            b64_str = data['b64']
            data = base64.b64decode(b64_str)
            
        return ImageOps.exif_transpose(Image.open(
            io.BytesIO(data)
        ))

    def inference(self, full_image):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.model.eval()
        with torch.no_grad():
            y = self.model(full_image, self.device)
        return y

    def postprocess(self, face = None, code = 0):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        
        if code != 0: 
            err_descrs = [
                'No face, too small or bad quality',
                'Found more than ONE face',
                'Face is not centered',
            ]   
            return [{
                'code': code,
                'description': err_descrs[code - 2],
                'age': None,
                'gender': None,
                'race': None,
            }]
        
        img = np.array(face)
        img = cv2.resize(img, self.INPUT_SIZE)
        img = img.astype(np.float32)
        img[..., 0] -= 103.939
        img[..., 1] -= 116.779
        img[..., 2] -= 123.68
        img = np.expand_dims(img, axis=0)
    
        results_ort = self.onnx_model.run(
            ["gender_pred/Sigmoid:0", 'age_pred/Softmax:0', 'ethnicity_pred/Softmax:0'], 
            {"input_1:0": img}
        )
        # results_ort = self.onnx_model.run(["emotion_preds"], {"x": img})[0]
        gender_score, age_score, race_score = results_ort
        gender_label = self.get_gender(gender_score)
        age_label = self.get_age(age_score)
        race_label = self.get_race(race_score)

        return [{
            'code': code,
            'description': 'Successful check',
            'age': age_label,
            'gender': gender_label,
            'race': race_label,
        }]
        

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        if self.onnx_model is None:
            self.load_model()
        
        full_image = self.preprocess(data)
        res = self.inference(full_image)
        if isinstance(res, int):
            return self.postprocess(code=res)
        return self.postprocess(face=res)
    
    def load_model(self):
        self.onnx_model = ort.InferenceSession("agre.onnx", providers=["CPUExecutionProvider"])

    def get_age(self, age_score: np.ndarray) -> int:
        age_score = age_score[0]
        idx_sorted = age_score.argsort()[::-1]
        age_score_sorted = age_score[idx_sorted]
        return int(((idx_sorted + 0.5) * age_score_sorted).sum())

    def get_race(self, race_score: np.ndarray) -> str:
        race_score = race_score[0]
        idx = race_score.argmax()
        return self.race_labels[idx]

    def get_gender(self, gender_score: np.ndarray) -> str:
        gender_score = gender_score[0][0]
        if gender_score >= 0.6:
            return 'Male'
        return 'Female'