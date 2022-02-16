from fastapi import FastAPI
import pickle
import numpy as np
import sklearn
from pydantic import  BaseModel
import uvicorn

def load_models():
    """
    load the models from disk
    and put them in a dictionary
    Returns:
        dict: loaded models
    """
    models = {
        "knn": pickle.load(open("./model_weights/clf.bin", 'rb'))
    }
    print("Models loaded from disk")
    return models

def get_prediction(feats, clf):
    x = feats
    y = clf.predict(x)[0]  # just get single value
    prob = clf.predict_proba(x)[0].tolist()  # send to list for return
    return {'prediction': int(round(y)), 'probability': prob}

# initiate API
app = FastAPI()

# define model for post request.
class ModelParams(BaseModel):
    feats: list
    # feat1: float
    # feat2: float
    # feat3: float
    # feat4: float
    # feat5: float
    # feat6: float
    # feat7: float
    # feat8: float

@app.post("/predict")
def predict(sample_feats: ModelParams):
    models = load_models()
    pred = get_prediction(sample_feats.feats, models['knn'])
    return pred

# if __name__ == "__main__":
#     uvicorn.run("hello_world_fastapi:app")
