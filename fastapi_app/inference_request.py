from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests

eb_service = True

if eb_service:
    host = "oil-palm-serving-env.eba-nq2vvvmx.eu-west-1.elasticbeanstalk.com"
else:
    host = "localhost:8000"

url = f"http://{host}/predict"

def make_request(img_name="no_oilpalm_class_0.jpg"):
    print(f"POST request for a {img_name}:")
    image = open(f"imgs/{img_name}", "rb")

    # MultipartEncoder takes care of the boundary string (automatically generates it)
    # The boundary string is required for the server to be able to parse the different parts of the request
    encoder = MultipartEncoder(fields={'file': (f'imgs/{img_name}', image, 'image/jpeg')})
    response = requests.post(url, data=encoder, headers={'Content-Type': encoder.content_type})

    print(response.json())

make_request()
make_request(img_name="with_oilpalm_class_1.jpg")

