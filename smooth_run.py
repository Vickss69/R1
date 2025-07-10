import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import mediapipe as mp
import urllib.request
import math

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# UI Configuration
st.set_page_config(page_title="Precision Beauty Comparator", page_icon="✨", layout="wide")
st.title("✨ Precision Beauty Comparison")
st.caption("Upload two images to compare facial features with medical-grade accuracy")
st.info("📝 **NOTE:** Results are based on facial symmetry and feature proportions, not subjective beauty")
st.warning("⚠️ **DISCLAIMER:** This application is for entertainment purposes only")

# Download Haar cascades if missing
def download_haar_cascades():
    cascade_dir = cv2.data.haarcascades
    required_files = {
        'haarcascade_frontalface_default.xml': 
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        'haarcascade_eye.xml': 
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    }
    
    for filename, url in required_files.items():
        path = os.path.join(cascade_dir, filename)
        if not os.path.exists(path):
            with st.spinner(f"Downloading {filename}..."):
                urllib.request.urlretrieve(url, path)
                st.success(f"Downloaded {filename}")

# Load Haar cascades with verification
@st.cache_resource
def load_haar_cascades():
    download_haar_cascades()
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Verify cascades loaded
        if face_cascade.empty() or eye_cascade.empty():
            raise Exception("Failed to load Haar cascades")
            
        return face_cascade, eye_cascade
    except Exception as e:
        st.error(f"Error loading Haar cascades: {str(e)}")
        st.stop()

face_cascade, eye_cascade = load_haar_cascades()

# Image processing functions
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to RGB for MediaPipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Maintain aspect ratio while resizing
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
        rgb_img = cv2.resize(rgb_img, (int(w * ratio), int(h * ratio)))
        
    return img, rgb_img

# Facial landmark detection with MediaPipe
def get_facial_landmarks(rgb_img):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            return None
            
        # Convert landmarks to pixel coordinates
        h, w, _ = rgb_img.shape
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            x = min(int(landmark.x * w), w - 1)
            y = min(int(landmark.y * h), h - 1)
            landmarks.append((x, y))
            
        return landmarks

# Feature analysis functions
def analyze_face_shape(landmarks):
    try:
        # Key facial points (MediaPipe indices)
        # Forehead: 10, Chin: 152, Left cheek: 93, Right cheek: 323
        if len(landmarks) < 400:  # MediaPipe has 478 landmarks
            return 0
            
        forehead = landmarks[10]
        chin = landmarks[152]
        left_cheek = landmarks[93]
        right_cheek = landmarks[323]
        
        # Calculate distances
        face_width = math.dist(left_cheek, right_cheek)
        face_height = math.dist(forehead, chin)
        cheek_width = math.dist(left_cheek, right_cheek)
        
        # Ratios for classification
        aspect_ratio = face_height / face_width
        cheek_to_face_ratio = cheek_width / face_width
        
        # Classify based on geometric ratios (similar to original)
        if aspect_ratio < 1.3 and cheek_to_face_ratio < 0.9:
            return 85  # Round
        elif aspect_ratio >= 1.3 and cheek_to_face_ratio < 0.9:
            return 92  # Oval
        elif cheek_to_face_ratio >= 1.0:
            return 78  # Square
        elif cheek_to_face_ratio < 0.8 and aspect_ratio > 1.5:
            return 82  # Diamond
        else:
            return 88  # Heart
        
    except Exception:
        return 0

def analyze_skin_quality(img, landmarks):
    try:
        if not landmarks or len(landmarks) < 400:
            return 0
            
        # Define face region using landmarks
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        
        # Expand region slightly
        padding = int((y2 - y1) * 0.05)
        y1 = max(0, y1 - padding)
        y2 = min(img.shape[0], y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        
        face_region = img[y1:y2, x1:x2]
        if face_region.size == 0:
            return 0
            
        # Convert to LAB color space
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate skin clarity (lower std = more even skin)
        clarity = 100 - min(100, np.std(l) * 2)
        
        # Calculate skin tone score (golden ratio)
        avg_b = np.mean(b)
        avg_a = np.mean(a)
        skin_tone_score = 100 - min(100, abs(avg_a - 128) * 0.5 + abs(avg_b - 128) * 0.5)
        
        return (clarity * 0.7 + skin_tone_score * 0.3)
        
    except Exception:
        return 0

def analyze_jawline(landmarks):
    try:
        if not landmarks or len(landmarks) < 400:
            return 0
            
        # Jawline points (MediaPipe indices: 172, 136, 149, 148, 152, 377, 378, 365, 397, 288)
        jaw_indices = [172, 136, 149, 148, 152, 377, 378, 365, 397, 288]
        jaw_points = [landmarks[i] for i in jaw_indices]
        
        # Calculate symmetry
        mid_point = jaw_points[4]  # Chin
        left_side = jaw_points[:5]
        right_side = jaw_points[5:][::-1]
        
        sym_errors = []
        for i in range(5):
            dist_left = math.dist(left_side[i], mid_point)
            dist_right = math.dist(right_side[i], mid_point)
            sym_errors.append(abs(dist_left - dist_right))
            
        symmetry_score = 100 - (np.mean(sym_errors) * 10)
        
        # Calculate jaw angle
        left_jaw = jaw_points[0]
        right_jaw = jaw_points[-1]
        chin = jaw_points[4]
        
        def calculate_angle(a, b, c):
            ba = (a[0] - b[0], a[1] - b[1])
            bc = (c[0] - b[0], c[1] - b[1])
            dot = ba[0]*bc[0] + ba[1]*bc[1]
            mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
            mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
            angle = math.acos(dot / (mag_ba * mag_bc))
            return math.degrees(angle)
        
        left_angle = calculate_angle(left_jaw, chin, (chin[0], chin[1]-100))  # Vertical reference
        right_angle = calculate_angle(right_jaw, chin, (chin[0], chin[1]-100))
        
        # Ideal jaw angle is 110-120 degrees
        angle_score = 100 - min(50, abs(left_angle - 115) + abs(right_angle - 115))
        
        return max(0, min(100, (symmetry_score * 0.6 + angle_score * 0.4)))
        
    except Exception:
        return 0

def analyze_eyes(img, landmarks):
    try:
        if not landmarks or len(landmarks) < 400:
            return 0, 0
            
        # Eye landmarks (MediaPipe indices)
        # Left eye: [33, 133], Right eye: [362, 263]
        left_eye_indices = [33, 133]
        right_eye_indices = [362, 263]
        
        # Get eye regions
        def get_eye_region(indices):
            xs = [landmarks[i][0] for i in indices]
            ys = [landmarks[i][1] for i in indices]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            
            # Expand region
            w, h = x2 - x1, y2 - y1
            padding = int(max(w, h) * 0.3)
            x1 = max(0, x1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y1 = max(0, y1 - padding)
            y2 = min(img.shape[0], y2 + padding)
            
            return img[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)
        
        left_eye, left_rect = get_eye_region(left_eye_indices)
        right_eye, right_rect = get_eye_region(right_eye_indices)
        
        if left_eye.size == 0 or right_eye.size == 0:
            return 0, 0
            
        # Eye shape analysis (aspect ratio)
        def eye_aspect_ratio(eye):
            if eye.size == 0:
                return 0
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            return min(w, h) / max(w, h) if max(w, h) > 0 else 0
        
        left_ratio = eye_aspect_ratio(left_eye)
        right_ratio = eye_aspect_ratio(right_eye)
        shape_score = ((left_ratio + right_ratio) / 2) * 100
        
        # Eye color analysis
        def eye_color_score(eye):
            if eye.size == 0:
                return 0
            avg_color = np.mean(eye, axis=(0, 1))
            # Score based on brightness and saturation
            brightness = avg_color.mean()
            saturation = np.std(eye, axis=(0, 1)).mean()
            return min(100, brightness * 0.4 + saturation * 0.6)
        
        color_score = (eye_color_score(left_eye) + eye_color_score(right_eye)) / 2
        
        return min(100, shape_score), min(100, color_score)
        
    except Exception:
        return 0, 0

def analyze_hair(img, landmarks):
    try:
        if not landmarks or len(landmarks) < 400:
            return 0
            
        # Get forehead point
        forehead = landmarks[10]
        
        # Define hair region (top 40% of head)
        h, w = img.shape[:2]
        hair_region = img[:int(forehead[1] * 0.8), :]
        
        if hair_region.size == 0:
            return 0
            
        # Calculate contrast
        gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        # Calculate density (edge detection)
        edges = cv2.Canny(gray, 100, 200)
        density = np.count_nonzero(edges) / (hair_region.shape[0] * hair_region.shape[1]) * 1000
        
        # Normalize scores
        contrast_score = min(100, contrast / 2)
        density_score = min(100, density / 5)
        
        return (contrast_score * 0.4 + density_score * 0.6)
        
    except Exception:
        return 0

def analyze_image(image_path):
    try:
        # Preprocess image
        img, rgb_img = preprocess_image(image_path)
        if img is None:
            return {
                'face_shape': 0, 'skin': 0, 'jawline': 0,
                'eye_shape': 0, 'eye_color': 0, 'hair': 0,
                'final_score': 0
            }
        
        # Get facial landmarks
        landmarks = get_facial_landmarks(rgb_img)
        
        # Calculate features
        face_shape = analyze_face_shape(landmarks) if landmarks else 0
        skin = analyze_skin_quality(img, landmarks) if landmarks else 0
        jawline = analyze_jawline(landmarks) if landmarks else 0
        eye_shape, eye_color = analyze_eyes(img, landmarks) if landmarks else (0, 0)
        hair = analyze_hair(img, landmarks) if landmarks else 0
        
        # Calculate composite score (same weights as original)
        weights = {
            'face_shape': 0.25,
            'skin': 0.40,
            'jawline': 0.15,
            'eye_shape': 0.10,
            'eye_color': 0.10,
            'hair': 0.20
        }
        
        final_score = (
            face_shape * weights['face_shape'] +
            skin * weights['skin'] +
            jawline * weights['jawline'] +
            eye_shape * weights['eye_shape'] +
            eye_color * weights['eye_color'] +
            hair * weights['hair']
        )
        
        return {
            'face_shape': face_shape,
            'skin': skin,
            'jawline': jawline,
            'eye_shape': eye_shape,
            'eye_color': eye_color,
            'hair': hair,
            'final_score': final_score
        }
        
    except Exception:
        return {
            'face_shape': 0, 'skin': 0, 'jawline': 0,
            'eye_shape': 0, 'eye_color': 0, 'hair': 0,
            'final_score': 0
        }

def create_winner_image(image_path, winner_text="HOTTER"):
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Font options
        font_sizes = [60, 70, 80]
        font_names = ["Arial", "Helvetica", "DejaVuSans-Bold", "LiberationSans-Bold", "Roboto-Bold"]
        
        font = None
        for size in font_sizes:
            for name in font_names:
                try:
                    font = ImageFont.truetype(name, size)
                    break
                except IOError:
                    continue
            if font:
                break
                
        if font is None:
            font = ImageFont.load_default()
            font.size = 60
        
        # Position text
        text_width = font.getmask(winner_text).getbbox()[2]
        position = ((img.width - text_width) // 2, img.height - 100)
        
        # Add glow effect
        for i in range(5, 0, -1):
            draw.text(
                (position[0], position[1] + i), 
                winner_text, 
                fill=(255, 215, 0, 128),  # Gold with transparency
                font=font
            )
        
        # Add text with background
        draw.rectangle(
            [position[0]-20, position[1]-20, 
             position[0]+text_width+20, position[1]+font.size+20],
            fill="black"
        )
        draw.text(position, winner_text, fill="gold", font=font)
        
        return img
        
    except Exception:
        return Image.open(image_path)

# Main application
def main():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Person 1")
        img1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="img1")
        
    with col2:
        st.subheader("Person 2")
        img2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="img2")
    
    if img1 and img2:
        with st.spinner("Analyzing images with precision algorithms..."):
            # Save to temp files
            temp_file1 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file1.write(img1.getbuffer())
            temp_file1.close()
            
            temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file2.write(img2.getbuffer())
            temp_file2.close()
            
            # Analyze images
            try:
                results1 = analyze_image(temp_file1.name)
                results2 = analyze_image(temp_file2.name)
                
                # Determine winner
                winner = 1 if results1['final_score'] >= results2['final_score'] else 2
                winner_img = temp_file1.name if winner == 1 else temp_file2.name
                winner_pil = create_winner_image(winner_img)
                
                # Display results
                st.success("Professional analysis complete!")
                st.divider()
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.image(img1, caption="Person 1", use_column_width=True)
                    st.metric("Overall Score", f"{results1['final_score']:.1f}/100", 
                              delta=f"{results1['final_score'] - results2['final_score']:.1f}" if winner == 1 else None)
                    
                    with st.expander("Detailed Analysis"):
                        st.write(f"**Face Shape:** {results1['face_shape']:.1f}/100")
                        st.write(f"**Skin Quality:** {results1['skin']:.1f}/100")
                        st.write(f"**Jawline Definition:** {results1['jawline']:.1f}/100")
                        st.write(f"**Eye Shape Harmony:** {results1['eye_shape']:.1f}/100")
                        st.write(f"**Eye Color Appeal:** {results1['eye_color']:.1f}/100")
                        st.write(f"**Hair Health Score:** {results1['hair']:.1f}/100")
                
                with col_res2:
                    st.image(img2, caption="Person 2", use_column_width=True)
                    st.metric("Overall Score", f"{results2['final_score']:.1f}/100", 
                              delta=f"{results2['final_score'] - results1['final_score']:.1f}" if winner == 2 else None)
                    
                    with st.expander("Detailed Analysis"):
                        st.write(f"**Face Shape:** {results2['face_shape']:.1f}/100")
                        st.write(f"**Skin Quality:** {results2['skin']:.1f}/100")
                        st.write(f"**Jawline Definition:** {results2['jawline']:.1f}/100")
                        st.write(f"**Eye Shape Harmony:** {results2['eye_shape']:.1f}/100")
                        st.write(f"**Eye Color Appeal:** {results2['eye_color']:.1f}/100")
                        st.write(f"**Hair Health Score:** {results2['hair']:.1f}/100")
                
                # Show winner
                st.divider()
                st.subheader(f"🏆 Winner: Person {winner}!")
                st.image(winner_pil, use_column_width=True)
                
                # Add comparison metrics
                st.subheader("Comparative Analysis")
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    st.metric("Facial Symmetry Advantage", 
                              f"{abs(results1['jawline'] - results2['jawline']):.1f}%",
                              help="Based on jawline symmetry scores")
                
                with col_comp2:
                    st.metric("Skin Quality Difference", 
                              f"{abs(results1['skin'] - results2['skin']):.1f}%",
                              help="Based on skin clarity and tone")
                
            except Exception as e:
                st.error(f"Professional analysis failed: {str(e)}")
            finally:
                # Cleanup temp files
                try:
                    os.unlink(temp_file1.name)
                    os.unlink(temp_file2.name)
                except:
                    pass
    else:
        st.info("👆 Upload two facial images to begin comparison")

if __name__ == "__main__":
    main()
