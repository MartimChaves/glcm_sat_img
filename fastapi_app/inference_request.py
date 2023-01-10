import requests

url = "http://localhost:8000/predict"

# Read the image file into memory
image_data = open("img_00000.jpg", "rb").read()

# Set the Content-Type header to "multipart/form-data"
headers = {"Content-Type": "multipart/form-data"}

# Create the POST data
data = {"image": ("img_00000.jpg", image_data)}

# Make the request
response = requests.post(url, headers=headers, data=data)

# Print the response
print(response.text)
