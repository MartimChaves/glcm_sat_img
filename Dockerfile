FROM python:3.10-bullseye


COPY . /app
WORKDIR /app

# This command is needed to install a library needed by opencv
# This library (libgl1) is related to graphics/images
RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install -r reqs.txt

WORKDIR /app/fastapi_app

EXPOSE 8000

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]