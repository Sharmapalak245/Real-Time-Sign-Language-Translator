import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time

# Fix for loading models saved on Unix on Windows
import pathlib
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

print("Starting ASL Detection System...")

try:
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    print("✓ YOLOv5 modules imported successfully")
    yolo_available = True
except ImportError as e:
    print(f"✗ YOLOv5 import error: {e}")
    yolo_available = False
# Load model with better error handling
if not yolo_available:
    print("✗ YOLOv5 not available, exiting...")
    sys.exit(1)

device = select_device('')
print(f"✓ Using device: {device}")

# Get the correct path to the model file
model_path = Path(__file__).parent / 'hehehehe.pt'
if not model_path.exists():
    print(f"✗ Model file not found at {model_path}")
    print("Please ensure 'hehehehe.pt' is in the same directory as this script")
    sys.exit(1)

print(f"✓ Found model at: {model_path}")

# Load model with simplified approach
try:
    print("Loading YOLOv5 model...")
    model = DetectMultiBackend(str(model_path), device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    print(f"✓ Model loaded successfully with {len(names)} classes")
    print(f"✓ Classes: {list(names.values())}")
    model.eval()  # Set to evaluation mode
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Initialize webcam with better error handling
print("Initializing webcam...")
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("✗ Could not open webcam on index 0. Trying other indices...")
    for i in range(1, 5):  # Try different camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✓ Webcam opened successfully on index {i}")
            break
    else:
        print("✗ No webcam found")
        sys.exit(1)
else:
    print("✓ Webcam opened successfully on index 0")

# Set webcam properties with error checking
try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Give camera time to initialize
    print("Waiting for camera to initialize...")
    time.sleep(2)
    
    # Test read a frame with retries
    for attempt in range(5):
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("✓ Webcam test successful")
            break
        print(f"Retry {attempt + 1}/5...")
        time.sleep(1)
    else:
        print("✗ Cannot read from webcam after multiple attempts")
        print("Trying to reinitialize webcam...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(0)
        time.sleep(2)
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("✗ Webcam initialization failed")
            sys.exit(1)
        print("✓ Webcam reinitialized successfully")
    
except Exception as e:
    print(f"✗ Error setting webcam properties: {e}")
    sys.exit(1)

# Get actual webcam properties
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print("=" * 50)
print("🎥 ASL SIGN DETECTION READY")
print("=" * 50)
print(f"✓ Resolution: {actual_width}x{actual_height}")
print(f"✓ FPS: {actual_fps}")
print(f"✓ Confidence: 0.75")
print("✓ Classes: Hello, Thanks, I love you, Please, Yes, No")
print("=" * 50)
print("📋 INSTRUCTIONS:")
print("• Make clear ASL signs in front of the camera")
print("• Keep hands well-lit and visible")
print("• Press 'q' to quit")
print("=" * 50)

# Detection parameters
detection_confidence = 0.45  # Increased confidence threshold
nms_threshold = 0.05

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("✗ Failed to read frame from webcam")
        break
    
    # Mirror effect for more natural interaction
    frame = cv2.flip(frame, 1)
    
    # Add status overlay
    cv2.putText(frame, "ASL Detection Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    try:
        # Prepare image for YOLOv5
        img = letterbox(frame, 640, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0  # Normalize to 0-1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # Add batch dimension
        img = img.to(device)

        # Run inference with error handling
        with torch.no_grad():
            pred = model(img, augment=False, visualize=False)
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=detection_confidence, iou_thres=nms_threshold)
        
        # Process detections
        if pred[0] is not None and len(pred[0]):
            det = pred[0]
            # Scale coordinates back to original frame size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # Color based on confidence
                if conf > 0.75:
                    color = (0, 255, 0)  # Green for high confidence
                else:
                    color = (0, 165, 255)  # Orange for lower confidence
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label with background
                label = f'{names[int(cls)]} {conf:.2f}'
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    except Exception as e:
        print(f"Detection error: {e}")
        # Continue without detection for this frame

    # Display the frame
    cv2.imshow("ASL Sign Detection", frame)

    # Check for quit command
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quit command received")
        break
    elif key == 27:  # ESC key
        print("ESC key pressed")
        break

print("Releasing webcam and closing windows...")
cap.release()
cv2.destroyAllWindows()
print("Application terminated successfully")