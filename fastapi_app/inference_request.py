import requests

eb_service = True

if eb_service:
    host = "oil-palm-serving-env.eba-nq2vvvmx.eu-west-1.elasticbeanstalk.com"
else:
    host = "127.0.0.1:8000"

url = f"http://{host}/predict"

def make_request(image_path="./imgs/no_oilpalm_class_0.jpg"):
    with open(image_path, "rb") as f:   
        files = {"file": f} 
        response = requests.post(url, files=files)  
        print(f"Img: {image_path.split('/')[2]}: \n"
              f"{response.json()}")

make_request()
make_request(image_path="./imgs/with_oilpalm_class_1.jpg")
