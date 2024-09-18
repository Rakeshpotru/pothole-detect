import os
import requests
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from ultralytics import YOLO
import numpy as np
import tempfile

app = Flask(__name__)
current_dir = os.path.dirname(__file__)

model_version = 'yolov8n'
model_paths = {
    'yolov8n': current_dir + '//content//runs//detect//train5//weights//best.pt',
    'yolov8m': current_dir + '//content//runs//detect//train5//weights//best.pt'
}
model_path = model_paths[model_version] 
model = YOLO(model_path)
font = ImageFont.load_default(56)
aream2 = 0

def calculate_focal_length(pixel_length, distance_to_object, image_width_pixels):
    # Using the formula: F = (P * D) / W
    focal_length = (pixel_length * distance_to_object) / image_width_pixels
    return focal_length

def calculate_real_world_length(focal_length, pixel_length, image_width_pixels, DISTANCE_TO_OBJECT):
    # Using the formula: W = (P * F) / D
    real_world_length = (pixel_length * focal_length) / DISTANCE_TO_OBJECT
    return real_world_length

def draw_detections(image, detections,shape):
    draw = ImageDraw.Draw(image)
    height, width =shape
    font = ImageFont.load_default(56)
    pixel_to_meter = 0.001
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf, cls_id = det[4:]
        
        DISTANCE_TO_OBJECT = 100  # Distance from camera to object in cm (0.9 meters)
        IMAGE_WIDTH_PIXELS = width  # Example image width in pixels
        FOCAL_LENGTH = None  # Focal length (to be calculated)  
        pixel_length = y2 - y1  # Length of the object in pixels
        # Calculate focal length based on the provided values
        FOCAL_LENGTH = calculate_focal_length(pixel_length, DISTANCE_TO_OBJECT, IMAGE_WIDTH_PIXELS)
        
        real_world_length = calculate_real_world_length(FOCAL_LENGTH, pixel_length, IMAGE_WIDTH_PIXELS,DISTANCE_TO_OBJECT)
        real_world_width = calculate_real_world_length(FOCAL_LENGTH, x2 - x1, IMAGE_WIDTH_PIXELS, DISTANCE_TO_OBJECT)
        
        area_cm2 = real_world_length * real_world_width
        area_m2 = area_cm2 /10000
        
        label = f'{int(cls_id)} {conf:.2f} | Area: {area_m2:.2f} m²'
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        text_width = font.getlength(label)
        text_height = font.getbbox(label)[3]  # Get the height from the bounding box
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height + 2], fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((x1, y1), label, font=font, fill=(255, 255, 255))
    
    return image

@app.route('/health', method=['GET'])
def chec_health():
    return "I am running fine"
    
@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.get_json()
    
    if 'image_url' not in data:
        return jsonify({"error": "'image_url' must be provided"}), 400
    
    image_url = data['image_url']
    
    # Download image from the URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise error if the URL is invalid
        img = Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error downloading image: {str(e)}"}), 400
    
    # Convert image to a format YOLO model can use (numpy array)
    img_np = np.array(img)
    
    # Perform YOLO detection on the image
    results = model(img_np, conf=0.2)  # Adjust confidence threshold as needed
    
    # Get detections (assuming the results format is consistent with this)
    detections = results[0].boxes.data.tolist()
    shape = 1280,720
    print("detections",detections)
    if detections:
        # Draw detections on the image
        annotated_image = draw_detections(img, detections,shape)
        
        # Save the image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        annotated_image.save(temp_file.name)
        
        # Return the image as a response
        return send_file(temp_file.name, mimetype='image/png', as_attachment=True, download_name="pothole_detections.png"),aream2
    
    return jsonify({
        "message": "No potholes detected."
    })

if __name__ == '__main__':
    app.run(debug=True)
