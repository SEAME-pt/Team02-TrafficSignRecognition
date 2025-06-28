import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from src.classificationNet import TinyTrafficSignNet
import time

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Class names for traffic signs
class_names = [
    "Speed 50km/h",      # 0
    "Speed 80km/h",      # 1  
    "Yield",             # 2
    "Stop",              # 3
    "Danger",            # 4
    "Crosswalk",         # 5
    "Unknown"            # 6
]

# Load the trained model
model = TinyTrafficSignNet(num_classes=7).to(device)
model.load_state_dict(torch.load('Models/traffic_signs/best_model_epoch_46.pth', map_location=device))
model.eval()

# Image preprocessing function
def preprocess_image(image, target_size=(30, 30)):
    """Preprocess image for traffic sign classification"""
    # Resize image to model input size
    img = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply same transforms as during training (use calculated mean/std)
    # You should use the actual mean/std calculated from your datasets
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Update with actual values
                                     std=[0.229, 0.224, 0.225])   # Update with actual values
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
    
    # Apply transforms
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    return img_tensor

def detect_and_classify_signs(frame, min_area=800, max_area=50000):
    """
    Detect potential traffic sign regions and classify them
    Enhanced criteria to reduce false positives
    """
    results = []
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # More restrictive color ranges for traffic signs
    # Red signs (stop, speed limits, etc.) - more restrictive
    red_lower1 = np.array([0, 70, 70])      # Increased saturation and value
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 70])    # Increased saturation and value
    red_upper2 = np.array([180, 255, 255])
    
    # Blue signs (crosswalk, etc.) - more restrictive
    blue_lower = np.array([100, 70, 70])    # Increased saturation and value
    blue_upper = np.array([130, 255, 255])
    
    # Yellow signs (warning, etc.) - more restrictive
    yellow_lower = np.array([20, 70, 70])   # Increased saturation and value
    yellow_upper = np.array([30, 255, 255])
    
    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Combine masks
    combined_mask = red_mask1 + red_mask2 + blue_mask + yellow_mask
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Enhanced area filtering (min and max)
        if min_area < area < max_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # More restrictive aspect ratio (traffic signs are more square)
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.4:  # More restrictive than 0.5-2.0
                
                # Additional geometric filters
                # Check if contour has reasonable number of vertices (not too jagged)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Signs usually have 3-8 vertices when approximated
                if 3 <= len(approx) <= 8:
                    
                    # Check if the region has sufficient color density
                    roi_mask = combined_mask[y:y+h, x:x+w]
                    color_density = np.sum(roi_mask > 0) / (w * h)
                    
                    # At least 30% of the region should be colored
                    if color_density > 0.3:
                        
                        # Extract the region
                        sign_region = frame[y:y+h, x:x+w]
                        
                        if sign_region.size > 0:
                            # Preprocess and classify
                            img_tensor = preprocess_image(sign_region)
                            
                            with torch.no_grad():
                                outputs = model(img_tensor)
                                probabilities = torch.softmax(outputs, dim=1)
                                confidence, predicted = torch.max(probabilities, 1)
                                
                                predicted_class = predicted.item()
                                confidence_score = confidence.item()
                                
                                # Enhanced filtering criteria
                                # 1. Higher confidence threshold
                                # 2. Don't show "Unknown" class (class 6)
                                if confidence_score > 0.75 and predicted_class != 6:  
                                    results.append({
                                        'bbox': (x, y, w, h),
                                        'class': predicted_class,
                                        'class_name': class_names[predicted_class],
                                        'confidence': confidence_score
                                    })
    
    return results

def draw_predictions(frame, predictions):
    """Draw bounding boxes and labels on the frame"""
    for pred in predictions:
        if pred['class'] == 6:
            continue

        x, y, w, h = pred['bbox']
        class_name = pred['class_name']
        confidence = pred['confidence']
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Background for text
        cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), (0, 255, 0), -1)
        
        # Text
        cv2.putText(frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame

# Test with video
def test_video(video_path):
    """Test the model on a video file"""
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and classify traffic signs
        predictions = detect_and_classify_signs(frame)
        
        # Draw predictions
        result_frame = draw_predictions(frame.copy(), predictions)
        
        # Display the result
        cv2.imshow("Traffic Sign Detection", result_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.05)  # Control frame rate
    
    cap.release()
    cv2.destroyAllWindows()

# Test with single image
def test_image(image_path):
    """Test the model on a single image"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Detect and classify traffic signs
    predictions = detect_and_classify_signs(frame)
    
    # Draw predictions
    result_frame = draw_predictions(frame.copy(), predictions)
    
    # Display the result
    cv2.imshow("Traffic Sign Detection", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print results
    print(f"Found {len(predictions)} traffic signs:")
    for i, pred in enumerate(predictions):
        print(f"  {i+1}: {pred['class_name']} (confidence: {pred['confidence']:.2f})")

if __name__ == "__main__":
    # Test with video
    print("Testing with video...")
    test_video("assets/video.mp4")
    
    # Uncomment to test with single image
    # test_image("path/to/your/test/image.jpg")
