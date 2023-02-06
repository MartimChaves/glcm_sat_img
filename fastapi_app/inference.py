from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from PIL import Image

import numpy as np
import pickle

from image_features import ImageFeatures


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

model = pickle.load(open("../model_weights/clf.bin", 'rb'))

def get_prediction(feats, clf):
    pred = clf.predict(feats)[0]  # just get single value
    prob = clf.predict_proba(feats)[0].tolist()  # send to list for return

    return_dict = {'prediction': int(round(pred)),
                   'probability': prob,
                   'features': feats}
    return return_dict

@app.post("/predict")
def predict(file: UploadFile):
    # Read Image
    img = Image.open(file.file)
    image_array = np.array(img)
    
    # Get Features
    img_features = ImageFeatures()
    features = img_features.calculate_features(image_array)
    normed_features = img_features.norm_features(features)

    # Use the model to generate a prediction
    prediction = get_prediction([list(normed_features)], model)
    
    # Return the prediction as a JSON response
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
