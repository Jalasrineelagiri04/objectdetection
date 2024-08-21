import torch
from PIL import Image
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Define function for object detection
def detect_objects(image_path):
    try:
        img = Image.open(image_path)
        results = model(img)

        # Extract detected objects and their bounding boxes
        objects = results.xyxy[0].cpu().numpy()

        return objects

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    image_path = Path("D:\objectdetection\yolov5\data\images")  # Replace with the actual path to your image
    objects = detect_objects(image_path)

    if objects is not None:
        if len(objects) > 0:
            print("Detected objects:")
            for obj in objects:
                print(obj)  # Print each detected object and its bounding box
        else:
            print("No objects detected in the image.")
    else:
        print("Object detection failed.")
