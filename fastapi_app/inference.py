from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from PIL import Image
from io import BytesIO

import numpy as np
import pickle
import cv2

import sys
sys.path.append("..")
from img_utils import Image_Funcs


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

model = pickle.load(open("../model_weights/clf.bin", 'rb'))


class ImageFeatures(Image_Funcs):
    
    def __init__(self):
        
        self.stats_calc = {
                'r_energy'     : self.get_glcm_metrics,
                'r_correlation': self.get_glcm_metrics,
                'r_contrast'   : self.get_glcm_metrics,
                'r_homogeneity': self.get_glcm_metrics,
                'g_energy'     : self.get_glcm_metrics,
                'h_correlation': self.get_glcm_metrics,
                's_correlation': self.get_glcm_metrics,
                's_contrast'   : self.get_glcm_metrics
            }

        # Number of features available
        self.n_feats = len(self.stats_calc.keys())
    
    def calculate_features(self, img):
        features = np.zeros((self.n_feats))

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        glcm_dict = {
            'r':self.get_glcm(img[...,0]),
            'g':self.get_glcm(img[...,1]),
            'h':self.get_glcm(img_hsv[...,0]),
            's':self.get_glcm(img_hsv[...,1])
        }
        
        for idx, (stat, calc_func) in enumerate(self.stats_calc.items()):
            channel = stat[0]
            channel_glcm = glcm_dict[channel]
            feat_val = calc_func(stat[2::],glcm=channel_glcm)
            features[idx] = feat_val
        
        return features

def get_prediction(feats, clf):
    pred = clf.predict(feats)[0]  # just get single value
    prob = clf.predict_proba(feats)[0].tolist()  # send to list for return

    return_dict = {'prediction': int(round(pred)),
                   'probability': prob,
                   'features': feats}
    return return_dict

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    image = read_imagefile(await file.read())
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Get Features
    img_features = ImageFeatures()
    features = img_features.calculate_features(image_array)

    # Use the model to generate a prediction
    prediction = get_prediction([list(features)], model)
    
    # Return the prediction as a JSON response
    return prediction

@app.post("/predict_v2/")
async def predict(file: UploadFile):
    # convert the image to numpy array
    img_bytes = await file.read()
    image_array = np.frombuffer(img_bytes, np.uint8)

    # Get Features
    img_features = ImageFeatures()
    features = img_features.calculate_features(image_array)

    # Use the model to generate a prediction
    prediction = get_prediction([list(features)], model)
    
    return {"prediction":prediction}

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
