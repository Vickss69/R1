import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import mediapipe as mp

st.info("📝 NOTE: The present comparison is only in the image, it's not in real life.")
st.warning("⚠️ DISCLAIMER: Don't misuse this application.")

# Initialize MediaPipe face detection and face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Set page title and description
st.title("Know who's more Beautiful")
st.write("Upload two images to compare and see which one scores higher!")

def calculate_face_shape(landmarks, image_width, image_height):
    try:
        # Extract key facial points
        forehead_left = (landmarks[10].x * image_width, landmarks[10].y * image_height)
        forehead_right = (landmarks[338].x * image_width, landmarks[338].y * image_height)
        cheek_left = (landmarks[454].x * image_width, landmarks[454].y * image_height)
        cheek_right = (landmarks[234].x * image_width, landmarks[234].y * image_height)
        chin = (landmarks[152].x * image_width, landmarks[152].y * image_height)

        # Calculate distances
        forehead_width = np.linalg.norm(np.array(forehead_left) - np.array(forehead_right))
        cheek_width = np.linalg.norm(np.array(cheek_left) - np.array(cheek_right))
        face_height = np.linalg.norm(np.array(chin) - np.array((
            (forehead_left[0] + forehead_right[0]) / 2,
            (forehead_left[1] + forehead_right[1]) / 2
        )))

        # Ratios for classification
        aspect_ratio = face_height / forehead_width
        cheek_to_jaw_ratio = cheek_width / forehead_width

        # Classification based on geometric ratios
        if aspect_ratio < 1.3 and cheek_to_jaw_ratio < 0.9:
            return 8
        elif aspect_ratio >= 1.3 and cheek_to_jaw_ratio < 0.9:
            return 18
        elif cheek_to_jaw_ratio >= 1.0:
            return 11
        elif cheek_to_jaw_ratio < 0.8 and aspect_ratio > 1.5:
            return 15
        elif cheek_width > forehead_width / 2 and chin[1] > cheek_left[1]:
            return 26
        elif cheek_width < forehead_width and aspect_ratio > 1.4:
            return 0
        else:
            return 22
    except Exception as e:
        st.error(f"Error in face shape calculation: {e}")
        return 0

def detect_face_shape(image_path):
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            image = cv2.imread(image_path)
            if image is None:
                return 0
                
            # Resize image
            height, width = image.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
                height, width = image.shape[:2]
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                return 0

            landmarks = results.multi_face_landmarks[0].landmark
            face_shape = calculate_face_shape(landmarks, width, height)
            return face_shape * 100 / 26
    except Exception as e:
        st.error(f"Error in face shape detection: {e}")
        return 0

def get_average_skin_color(image_path):
    try:
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            image = cv2.imread(image_path)
            if image is None:
                return 0
                
            # Resize image
            height, width = image.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
                height, width = image.shape[:2]
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = face_detection.process(rgb_image)
            if not results.detections:
                return 0
            
            # Extract face regions
            skin_pixels = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * width), int(bbox.ymin * height), \
                            int(bbox.width * width), int(bbox.height * height)
                
                # Adjust coordinates
                x, y = max(0, x), max(0, y)
                w, h = min(width - x, w), min(height - y, h)
                
                if w <= 0 or h <= 0:
                    continue
                    
                face_roi = image[y:y+h, x:x+w]
                
                # Sample pixels
                step = max(1, min(h, w) // 20)
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        skin_pixels.append(face_roi[i, j])
            
            if not skin_pixels:
                return 0
                
            # Compute average
            skin_pixels = np.array(skin_pixels)
            avg_bgr = np.mean(skin_pixels, axis=0)
            a, b, c = avg_bgr
            return np.sqrt(a**2 + b**2 + c**2)
    except Exception as e:
        st.error(f"Error in skin color detection: {e}")
        return 0

def detect_eyes_shape(image_path):
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            image = cv2.imread(image_path)
            if image is None:
                return 0
                
            # Resize image
            height, width = image.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
                height, width = image.shape[:2]
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face mesh
            results = face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                return 0

            landmarks = results.multi_face_landmarks[0].landmark
            
            # Eye indices for MediaPipe
            LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153]
            RIGHT_EYE_INDICES = [362, 385, 387, 380, 373, 374]
            
            def get_eye_aspect_ratio(eye_indices):
                points = []
                for idx in eye_indices:
                    x = landmarks[idx].x * width
                    y = landmarks[idx].y * height
                    points.append((x, y))
                    
                # Calculate eye aspect ratio
                width_eye = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
                height_eye = (np.linalg.norm(np.array(points[1]) - np.array(points[4])) + 
                            np.linalg.norm(np.array(points[2]) - np.array(points[5]))) / 2
                return height_eye / max(width_eye, 1e-6)  # Avoid division by zero

            left_eye_ratio = get_eye_aspect_ratio(LEFT_EYE_INDICES)
            right_eye_ratio = get_eye_aspect_ratio(RIGHT_EYE_INDICES)
            
            # Calculate shape score
            def shape_score(ratio):
                if ratio < 0.25: return 38
                elif ratio <= 0.35: return 28
                else: return 22
            
            left_score = shape_score(left_eye_ratio)
            right_score = shape_score(right_eye_ratio)
            
            return ((left_score + right_score) * 100) / 72
    except Exception as e:
        st.error(f"Error in eye shape detection: {e}")
        return 0

def detect_eye_colors(image_path):
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            image = cv2.imread(image_path)
            if image is None:
                return 0
                
            # Resize image
            height, width = image.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
                height, width = image.shape[:2]
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face mesh
            results = face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                return 0

            landmarks = results.multi_face_landmarks[0].landmark
            
            # Eye indices
            LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153]
            RIGHT_EYE_INDICES = [362, 385, 387, 380, 373, 374]
            
            def get_eye_region(indices):
                points = []
                for idx in indices:
                    x = int(landmarks[idx].x * width)
                    y = int(landmarks[idx].y * height)
                    points.append((x, y))
                return points
            
            left_eye_points = get_eye_region(LEFT_EYE_INDICES)
            right_eye_points = get_eye_region(RIGHT_EYE_INDICES)
            
            # Create masks for eye regions
            def extract_eye_region(points):
                mask = np.zeros((height, width), dtype=np.uint8)
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                eye_region = cv2.bitwise_and(image, image, mask=mask)
                return eye_region[np.where(mask == 255)]
            
            left_eye_pixels = extract_eye_region(left_eye_points)
            right_eye_pixels = extract_eye_region(right_eye_points)
            
            # Classify eye color
            def classify_eye_color(pixels):
                if len(pixels) == 0:
                    return 15
                avg_color = np.mean(pixels, axis=0)
                b, g, r = avg_color
                if r > 100 and g < 70 and b < 40: return 5
                elif r > 140 and g > 100 and b < 60: return 19
                elif r < 100 and g < 100 and b > 120: return 29
                elif r < 100 and g > 120 and b < 100: return 14
                elif r > 100 and g > 80 and b < 60: return 9
                elif r < 100 and g < 100 and b < 80: return 24
                else: return 15
            
            left_score = classify_eye_color(left_eye_pixels)
            right_score = classify_eye_color(right_eye_pixels)
            return ((left_score + right_score) / 2) * 100 / 30
    except Exception as e:
        st.error(f"Error in eye color detection: {e}")
        return 0

def calculate_final_hair_score(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
            
        # Resize image
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
            height, width = image.shape[:2]
            
        # Focus on top 40% of image
        hair_region = image[:int(height * 0.4), :]
        
        if hair_region.size == 0:
            return 0
            
        # Calculate average color
        avg_color = np.mean(hair_region, axis=(0, 1))
        b, g, r = avg_color
        color_score = 100 - (np.sqrt(r**2 + g**2 + b**2) / 441.67) * 100
        
        # Calculate hair density (using edge detection)
        gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        hair_density = np.count_nonzero(edges) / edges.size * 100
        
        # Final hair score
        return (color_score + hair_density) / 2
    except Exception as e:
        st.error(f"Error in hair score calculation: {e}")
        return 0

def mark_winner(image_path, is_winner=True):
    try:
        image = Image.open(image_path)
        if not is_winner:
            return image

        draw = ImageDraw.Draw(image)
        text = "Hott ONE"
        
        try:
            font = ImageFont.truetype("arial.ttf", 50)
        except:
            font = ImageFont.load_default()
            
        # Calculate text position
        img_width, img_height = image.size
        try:
            # For newer Pillow versions
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # For older Pillow versions
            text_width, text_height = draw.textsize(text, font=font)
            
        x = (img_width - text_width) // 2
        y = img_height - text_height - 20
        
        # Draw background and text
        draw.rectangle([x-10, y-10, x+text_width+10, y+text_height+10], fill="white")
        draw.text((x, y), text, fill="black", font=font)
        
        return image
    except Exception as e:
        st.error(f"Error marking winner: {e}")
        return Image.open(image_path)

def analyze_image(image_path, progress_callback=None):
    try:
        metrics = {}
        
        if progress_callback:
            progress_callback(0, "Detecting face shape...")
        metrics['face_shape'] = detect_face_shape(image_path)
        
        # If no face detected, return default values
        if metrics['face_shape'] == 0:
            for key in ['skin_color', 'skin_score', 'eye_shape', 'eye_color', 'hair_score']:
                metrics[key] = 0
            metrics['final_score'] = 0
            if progress_callback:
                progress_callback(100, "No face detected!")
            return metrics
        
        if progress_callback:
            progress_callback(20, "Analyzing skin color...")
        metrics['skin_color'] = get_average_skin_color(image_path)
        metrics['skin_score'] = (1.5 * 100 * metrics['skin_color'] / (256 * np.sqrt(3)))
        
        if progress_callback:
            progress_callback(50, "Analyzing eye shape...")
        metrics['eye_shape'] = detect_eyes_shape(image_path)
        
        if progress_callback:
            progress_callback(65, "Analyzing eye color...")
        metrics['eye_color'] = detect_eye_colors(image_path)
        
        if progress_callback:
            progress_callback(80, "Analyzing hair...")
        metrics['hair_score'] = calculate_final_hair_score(image_path)
        
        # Calculate composite score
        metrics['final_score'] = (
            metrics['face_shape'] * 25 + 
            metrics['skin_score'] * 40 + 
            metrics['eye_shape'] * 10 + 
            metrics['eye_color'] * 10 + 
            metrics['hair_score'] * 20
        ) / 100
        
        if progress_callback:
            progress_callback(100, "Analysis complete!")
            
        return metrics
    except Exception as e:
        st.error(f"Error in image analysis: {e}")
        return {
            'face_shape': 0, 'skin_color': 0, 'skin_score': 0,
            'eye_shape': 0, 'eye_color': 0, 'hair_score': 0, 'final_score': 0
        }

# File uploader for the two images
col1, col2 = st.columns(2)

with col1:
    st.subheader("Image 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="file1")

with col2:
    st.subheader("Image 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="file2")

# Process images when both are uploaded
if uploaded_file1 is not None and uploaded_file2 is not None:
    st.write("Processing images... This may take a moment.")
    
    # Save the uploaded files to temporary locations
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file1:
        temp_file1.write(uploaded_file1.getvalue())
        temp_file1_path = temp_file1.name
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file2:
        temp_file2.write(uploaded_file2.getvalue())
        temp_file2_path = temp_file2.name
    
    # Progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Process first image
        def update_progress(percent, message=""):
            progress_bar.progress(percent)
            if message:
                status_text.text(message)
        
        update_progress(0, "Analyzing first image...")
        metrics1 = analyze_image(temp_file1_path, 
            lambda p, msg: update_progress(p/2, f"Image 1: {msg}")
        )
        
        # Process second image
        update_progress(50, "Analyzing second image...")
        metrics2 = analyze_image(temp_file2_path, 
            lambda p, msg: update_progress(50 + p/2, f"Image 2: {msg}")
        )
        
        update_progress(100, "Comparing results...")
        
        # Extract scores
        s1 = metrics1['final_score']
        s2 = metrics2['final_score']
        
        # Determine winner
        winner_path = temp_file1_path if s1 >= s2 else temp_file2_path
        winner_pil = mark_winner(winner_path, True)
        
        # Display results
        status_text.empty()
        st.subheader("Results")
        col3, col4 = st.columns(2)
        
        with col3:
            st.image(uploaded_file1, caption="Image 1", use_column_width=True)
            st.metric("Score", f"{s1:.2f}")
            
            with st.expander("Detailed Scores for Image 1"):
                st.write(f"Face Shape: {metrics1['face_shape']:.2f}")
                st.write(f"Skin Score: {metrics1['skin_score']:.2f}")
                st.write(f"Eye Shape: {metrics1['eye_shape']:.2f}")
                st.write(f"Eye Color: {metrics1['eye_color']:.2f}")
                st.write(f"Hair Score: {metrics1['hair_score']:.2f}")
        
        with col4:
            st.image(uploaded_file2, caption="Image 2", use_column_width=True) 
            st.metric("Score", f"{s2:.2f}")
            
            with st.expander("Detailed Scores for Image 2"):
                st.write(f"Face Shape: {metrics2['face_shape']:.2f}")
                st.write(f"Skin Score: {metrics2['skin_score']:.2f}")
                st.write(f"Eye Shape: {metrics2['eye_shape']:.2f}")
                st.write(f"Eye Color: {metrics2['eye_color']:.2f}")
                st.write(f"Hair Score: {metrics2['hair_score']:.2f}")
        
        # Display the winner
        st.subheader("Winner")
        winner_text = "Image 1 Wins!" if s1 >= s2 else "Image 2 Wins!"
        st.image(winner_pil, caption=winner_text)
            
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_file1_path)
            os.unlink(temp_file2_path)
        except:
            pass
else:
    st.info("Please upload both images to compare.")
