from fastapi import FastAPI
import pickle
from SatImages import SatImage
import uvicorn

def load_models():
    """
    load the models from disk
    and put them in a dictionary
    Returns:
        dict: loaded models
    """
    models = {
        "knn": pickle.load(open("../model_weights/clf.bin", 'rb'))
    }
    print("Models loaded from disk")
    return models

def get_prediction(feats, clf):
    pred = clf.predict(feats)[0]  # just get single value
    prob = clf.predict_proba(feats)[0].tolist()  # send to list for return
    return {'prediction': int(round(pred)), 'probability': prob}

# initiate API
app = FastAPI()

@app.get("/")
def index():
    return {'message':'hello, everyone'}

@app.post("/predict")
def predict(data: SatImage):
    data = data.dict()
    feats = [[feat_val for _, feat_val in data.items()]]
    models = load_models()
    pred = get_prediction(feats, models['knn'])
    return pred

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
