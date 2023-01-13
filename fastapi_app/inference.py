from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from PIL import Image
from io import BytesIO

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
    normed_features = img_features.norm_features(features)

    # Use the model to generate a prediction
    prediction = get_prediction([list(normed_features)], model)
    
    # Return the prediction as a JSON response
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
