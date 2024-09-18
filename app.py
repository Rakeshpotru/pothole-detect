import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from ultralytics import YOLO
import numpy as np
import tempfile
from pydantic import BaseModel

app = FastAPI()
current_dir = os.path.dirname(__file__)

model_version = 'yolov8n'
model_paths = {
    'yolov8n': os.path.join(current_dir, 'content', 'runs', 'detect', 'train5', 'weights', 'best.pt'),
    'yolov8m': os.path.join(current_dir, 'content', 'runs', 'detect', 'train5', 'weights', 'best.pt')
}
model_path = model_paths[model_version]
model = YOLO(model_path)
font = ImageFont.load_default(56)

class ImageRequest(BaseModel):
    image_url: str

def calculate_focal_length(pixel_length, distance_to_object, image_width_pixels):
    return (pixel_length * distance_to_object) / image_width_pixels

def calculate_real_world_length(focal_length, pixel_length, image_width_pixels, distance_to_object):
    return (pixel_length * focal_length) / distance_to_object

def draw_detections(image, detections, shape):
    draw = ImageDraw.Draw(image)
    height, width = shape
    pixel_to_meter = 0.001

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf, cls_id = det[4:]

        DISTANCE_TO_OBJECT = 100  # Distance from camera to object in cm
        IMAGE_WIDTH_PIXELS = width
        pixel_length = y2 - y1

        FOCAL_LENGTH = calculate_focal_length(pixel_length, DISTANCE_TO_OBJECT, IMAGE_WIDTH_PIXELS)
        real_world_length = calculate_real_world_length(FOCAL_LENGTH, pixel_length, IMAGE_WIDTH_PIXELS, DISTANCE_TO_OBJECT)
        real_world_width = calculate_real_world_length(FOCAL_LENGTH, x2 - x1, IMAGE_WIDTH_PIXELS, DISTANCE_TO_OBJECT)

        area_cm2 = real_world_length * real_world_width
        area_m2 = area_cm2 / 10000

        label = f'{int(cls_id)} {conf:.2f} | Area: {area_m2:.2f} m²'
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        text_width = font.getlength(label)
        text_height = font.getbbox(label)[3]
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height + 2], fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((x1, y1), label, font=font, fill=(255, 255, 255))

    return image

@app.get('/health')
def check_health():
    return {"status": "I am running fine"}

@app.post('/upload')
async def upload_file(image_request: ImageRequest):
    image_url = image_request.image_url

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

    img_np = np.array(img)
    results = model(img_np, conf=0.2)
    detections = results[0].boxes.data.tolist()
    shape = 1280, 720

    if detections:
        annotated_image = draw_detections(img, detections, shape)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        annotated_image.save(temp_file.name)

        return FileResponse(temp_file.name, media_type='image/png', filename="pothole_detections.png")

    return {"message": "No potholes detected."}
