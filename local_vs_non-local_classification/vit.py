import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import mediapipe as mp
import logging
import time
import uuid
import torch
from torchvision import transforms
from flask_cors import CORS

# Import the model
from model import VisionTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins

# Define constants
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_VIDEO_FRAMES = 10  # Number of frames to sample from video
MODEL_PATH = "pretrained_vit_trained_model.pth"  # UPDATE THIS with your model path

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Define image transformation for ViT input (matches torchvision ViT_B_16 preprocessing)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ViT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    try:
        # Initialize the VisionTransformer model
        model = VisionTransformer(num_classes=2)
        
        # Check if the model file exists
        if not os.path.isfile(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return None
            
        # Load the pre-trained weights with more explicit error handling
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            # Debug info about checkpoint
            logger.info(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'not a dict'}")
            
            # Load the model state
            model.load_from_checkpoint(MODEL_PATH)
            model.to(device)
            model.eval()
            
            logger.info(f"Model loaded successfully on {device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model weights: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Load the model
model = load_model()

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def detect_face_from_image(image):
    """Detect face from image using MediaPipe and return face image"""
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image input")
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Process the image with MediaPipe Face Detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image_rgb)
            
            if not results.detections:
                logger.info("No face detected in the image")
                return None  # No face detected
            
            # Get the first face detected (with highest confidence)
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Convert normalized coordinates to pixel values
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within image boundaries
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            # Extract face image
            face_img = image[y:y+height, x:x+width]
            
            return face_img, (x, y, width, height)
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return None

def encode_image_to_base64(image):
    """Convert image to base64 for sending to frontend"""
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            return None
            
        # Resize image to reduce payload size
        max_dim = 300
        h, w = image.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', image)
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return None

def classify_face_with_vit(face_img):
    """Use trained ViT model to classify face as local or non-local"""
    try:
        if model is None:
            logger.error("Model not loaded")
            return {
                "classification": "non-local",  # Default
                "confidence": 50.0,
                "error": "Model not loaded"
            }
            
        # Preprocess the image
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(face_img_rgb)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # Get prediction from model
        with torch.no_grad():
            outputs = model(input_batch)
            
            # Apply softmax 
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probability_values = probabilities[0].cpu().numpy()
            
            # Assuming index 0 is "local" and index 1 is "non-local"
            local_probability = probability_values[0] * 100
            non_local_probability = probability_values[1] * 100
            
            classification = "local" if local_probability > non_local_probability else "non-local"
            confidence = float(max(local_probability, non_local_probability))
            
            return {
                "classification": classification,
                "confidence": confidence,
                "probabilities": {
                    "local": float(local_probability),
                    "non_local": float(non_local_probability)
                }
            }
    except Exception as e:
        logger.error(f"Error in face classification with ViT: {str(e)}")
        return {
            "classification": "non-local",  # Default to non-local on error
            "confidence": 50.0,
            "error": str(e)
        }

def process_video(video_path):
    """Process video, extract frames, detect and classify faces"""
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return {"error": "Failed to open video file"}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video opened: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Determine frame interval for processing
        interval = max(1, total_frames // MAX_VIDEO_FRAMES)
        
        results = []
        frame_count = 0
        processed_frames = 0
        frames_with_faces = 0
        
        # Process video frames
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while cap.isOpened() and processed_frames < MAX_VIDEO_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process only at intervals
                if frame_count % interval != 0:
                    frame_count += 1
                    continue
                
                # Detect face in the frame
                processed_frames += 1
                face_result = detect_face_from_image(frame)
                
                if face_result:
                    frames_with_faces += 1
                    face_img, face_coords = face_result
                    
                    # Classify face using ViT model
                    classification_result = classify_face_with_vit(face_img)
                    
                    # Add frame information and face image
                    classification_result["frame_number"] = frame_count
                    classification_result["timestamp"] = frame_count / fps
                    classification_result["face_coordinates"] = face_coords
                    classification_result["face_image"] = encode_image_to_base64(face_img)
                    
                    results.append(classification_result)
                
                frame_count += 1
        
        # Release the video file
        cap.release()
        
        if not results:
            return {"error": "No faces detected in the video"}
        
        # Determine the most common classification
        local_count = sum(1 for r in results if r["classification"] == "local")
        non_local_count = len(results) - local_count
        
        overall_classification = "local" if local_count > non_local_count else "non-local"
        confidence = (max(local_count, non_local_count) / len(results)) * 100
        
        return {
            "classification": overall_classification,
            "confidence": confidence,
            "frames_processed": processed_frames,
            "frames_with_faces": frames_with_faces,
            "face_detections": results[:5],  # Limit to 5 face results to reduce payload size
            "total_detections": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {"error": f"Error processing video: {str(e)}"}
    finally:
        # Make sure to release the video capture
        if 'cap' in locals() and cap is not None:
            cap.release()

# Simple health check endpoint
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "healthy", "timestamp": time.time()})

# Main route for the application homepage
@app.route('/', methods=['GET'])
def index():
    return '''
    <html>
        <head>
            <title>Face Classification API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #3a86ff; }
                code { background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Face Classification API</h1>
            <p>Use the <code>/classify</code> endpoint to upload an image or video for face detection and classification.</p>
            <p>Example usage: POST a file to <code>http://127.0.0.1:5000/classify</code></p>
        </body>
    </html>
    '''

# Classification endpoint
@app.route('/classify', methods=['POST'])
def classify():
    start_time = time.time()
    logger.info("Received classification request")
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Generate a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        if allowed_image_file(file.filename):
            logger.info(f"Processing image: {file.filename}")
            
            # Process image
            image = cv2.imread(file_path)
            if image is None:
                logger.error("Failed to read image file")
                os.remove(file_path)
                return jsonify({"error": "Failed to read image file"}), 400
            
            # Detect face
            logger.info("Detecting face")
            result = detect_face_from_image(image)
            
            if result is None:
                logger.error("No face detected in the image")
                os.remove(file_path)
                return jsonify({"error": "No face detected in the image"}), 400
            
            face_img, _ = result
            
            # Classify face using ViT model
            logger.info("Classifying face using ViT model")
            classification_result = classify_face_with_vit(face_img)
            
            # Add base64 image data for UI display
            face_image_base64 = encode_image_to_base64(face_img)
            if face_image_base64:
                classification_result["face_image"] = face_image_base64
            
            classification_result["file_type"] = "image"
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            logger.info(f"Image classification completed in {time.time() - start_time:.2f} seconds")
            return jsonify(classification_result)
            
        elif allowed_video_file(file.filename):
            logger.info(f"Processing video: {file.filename}")
            
            # Process video
            video_result = process_video(file_path)
            
            if "error" in video_result:
                logger.error(f"Video processing error: {video_result['error']}")
                os.remove(file_path)
                return jsonify(video_result), 400
            
            video_result["file_type"] = "video"
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            logger.info(f"Video classification completed in {time.time() - start_time:.2f} seconds")
            return jsonify(video_result)
            
        else:
            logger.error("File type not allowed")
            os.remove(file_path)
            return jsonify({"error": "File type not allowed"}), 400
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Ensure clean up
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        
        return jsonify({
            "error": "Server error processing your request. Please try again.",
            "status": "Error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Face Classification API")
    app.run(debug=True, threaded=True)
    print(f"Looking for model at absolute path: {os.path.abspath(MODEL_PATH)}")