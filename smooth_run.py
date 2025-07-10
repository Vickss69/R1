# beauti_compare.py
import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import time
import urllib.request

# Preload models at startup
@st.cache_resource
def load_models():
    # Download face landmark model if missing
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        with st.spinner("Downloading facial landmark model (60MB)... This may take a minute"):
            model_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
            urllib.request.urlretrieve(model_url, predictor_path)
    
    # Load models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    return detector, predictor, face_cascade, eye_cascade

# UI Setup
st.set_page_config(page_title="Beauty Comparator", page_icon="✨", layout="wide")
st.title("✨ Know Who's More Beautiful")
st.caption("Upload two images to compare facial features and see which scores higher!")
st.info("📝 **NOTE:** This comparison is based on image analysis only, not real life")
st.warning("⚠️ **DISCLAIMER:** This application is for entertainment purposes only")

# Load models
try:
    detector, predictor, face_cascade, eye_cascade = load_models()
except Exception as e:
    st.error(f"❌ Model loading failed: {str(e)}")
    st.stop()

# Image processing functions
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Maintain aspect ratio while resizing
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    return img

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return detector(gray)

def get_facial_landmarks(gray, face):
    return predictor(gray, face)

# Feature analysis functions
def analyze_face_shape(image_path):
    try:
        img = preprocess_image(image_path)
        if img is None:
            return 0
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(img)
        if not faces:
            return 0
            
        face = faces[0]
        landmarks = get_facial_landmarks(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Calculate face ratios
        jaw_width = np.linalg.norm(points[16] - points[0])
        face_height = np.linalg.norm(points[8] - points[27])
        cheek_width = np.linalg.norm(points[13] - points[3])
        
        aspect_ratio = face_height / jaw_width
        cheek_ratio = cheek_width / jaw_width
        
        # Classify face shape
        if aspect_ratio < 1.3 and cheek_ratio < 0.9:
            return 8
        elif aspect_ratio >= 1.3 and cheek_ratio < 0.9:
            return 18
        elif cheek_ratio >= 1.0:
            return 11
        elif cheek_ratio < 0.8 and aspect_ratio > 1.5:
            return 15
        else:
            return 22
            
    except Exception:
        return 0

def analyze_skin_quality(image_path):
    try:
        img = preprocess_image(image_path)
        if img is None:
            return 0
            
        # Focus on center face region
        h, w = img.shape[:2]
        y1, y2 = int(h*0.2), int(h*0.8)
        x1, x2 = int(w*0.25), int(w*0.75)
        face_region = img[y1:y2, x1:x2]
        
        # Convert to LAB color space
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate skin clarity (lower std = more even skin)
        clarity = 100 - min(100, np.std(l) * 2)
        return max(0, clarity)
        
    except Exception:
        return 0

def analyze_jawline(image_path):
    try:
        img = preprocess_image(image_path)
        if img is None:
            return 0
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(img)
        if not faces:
            return 0
            
        face = faces[0]
        landmarks = get_facial_landmarks(gray, face)
        jaw_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)])
        
        # Calculate symmetry
        mid_point = jaw_points[8]
        left_side = jaw_points[:8]
        right_side = jaw_points[8:][::-1]
        
        sym_errors = []
        for i in range(8):
            dist_left = np.linalg.norm(left_side[i] - mid_point)
            dist_right = np.linalg.norm(right_side[i] - mid_point)
            sym_errors.append(abs(dist_left - dist_right))
            
        symmetry = 100 - (np.mean(sym_errors) / 10)
        return max(0, min(100, symmetry))
        
    except Exception:
        return 0

def analyze_eyes(image_path):
    try:
        img = preprocess_image(image_path)
        if img is None:
            return 0, 0
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(img)
        if not faces:
            return 0, 0
            
        face = faces[0]
        landmarks = get_facial_landmarks(gray, face)
        
        # Eye shape analysis
        def eye_aspect_ratio(eye_points):
            vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
            vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
            horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
            return (vertical1 + vertical2) / (2.0 * horizontal)
        
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        left_ratio = eye_aspect_ratio(left_eye)
        right_ratio = eye_aspect_ratio(right_eye)
        shape_score = ((left_ratio + right_ratio) / 2) * 100
        
        # Eye color analysis (simplified)
        eye_region = img[min(left_eye[:,1]):max(left_eye[:,1]), 
                         min(left_eye[:,0]):max(left_eye[:,0])]
        avg_color = np.mean(eye_region, axis=(0,1))
        color_score = np.mean(avg_color) / 2.55  # Convert 0-255 to 0-100
        
        return min(100, shape_score), min(100, color_score)
        
    except Exception:
        return 0, 0

def analyze_hair(image_path):
    try:
        img = preprocess_image(image_path)
        if img is None:
            return 0
            
        # Focus on top portion of image
        hair_region = img[:int(img.shape[0]*0.4), :]
        
        # Calculate contrast between hair and skin
        gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        # Normalize contrast score (0-100)
        return min(100, contrast / 1.5)
        
    except Exception:
        return 0

def analyze_image(image_path):
    face_shape = analyze_face_shape(image_path) * 100 / 26
    skin = analyze_skin_quality(image_path)
    jawline = analyze_jawline(image_path)
    eye_shape, eye_color = analyze_eyes(image_path)
    hair = analyze_hair(image_path)
    
    # Weighted composite score
    weights = {
        'face_shape': 0.25,
        'skin': 0.30,
        'jawline': 0.15,
        'eye_shape': 0.10,
        'eye_color': 0.10,
        'hair': 0.10
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

def create_winner_image(image_path, winner_text="HOTTER"):
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try multiple font options
        font = None
        font_sizes = [40, 50, 60]
        font_names = [
            "arial.ttf", 
            "DejaVuSans.ttf", 
            "LiberationSans-Bold.ttf",
            "Roboto-Bold.ttf"
        ]
        
        for size in font_sizes:
            for font_name in font_names:
                try:
                    font = ImageFont.truetype(font_name, size)
                    break
                except IOError:
                    continue
            if font:
                break
                
        if font is None:
            font = ImageFont.load_default(size=40)
        
        # Position text
        text_width = font.getmask(winner_text).getbbox()[2]
        position = ((img.width - text_width) // 2, img.height - 80)
        
        # Add text with background
        draw.rectangle(
            [position[0]-10, position[1]-10, 
             position[0]+text_width+10, position[1]+50],
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
        with st.spinner("Analyzing images..."):
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
                st.success("Analysis complete!")
                st.divider()
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.image(img1, caption="Person 1", use_column_width=True)
                    st.metric("Overall Score", f"{results1['final_score']:.1f}/100")
                    
                    with st.expander("Detailed Analysis"):
                        st.write(f"**Face Shape:** {results1['face_shape']:.1f}/100")
                        st.write(f"**Skin Quality:** {results1['skin']:.1f}/100")
                        st.write(f"**Jawline:** {results1['jawline']:.1f}/100")
                        st.write(f"**Eye Shape:** {results1['eye_shape']:.1f}/100")
                        st.write(f"**Eye Color:** {results1['eye_color']:.1f}/100")
                        st.write(f"**Hair Score:** {results1['hair']:.1f}/100")
                
                with col_res2:
                    st.image(img2, caption="Person 2", use_column_width=True)
                    st.metric("Overall Score", f"{results2['final_score']:.1f}/100")
                    
                    with st.expander("Detailed Analysis"):
                        st.write(f"**Face Shape:** {results2['face_shape']:.1f}/100")
                        st.write(f"**Skin Quality:** {results2['skin']:.1f}/100")
                        st.write(f"**Jawline:** {results2['jawline']:.1f}/100")
                        st.write(f"**Eye Shape:** {results2['eye_shape']:.1f}/100")
                        st.write(f"**Eye Color:** {results2['eye_color']:.1f}/100")
                        st.write(f"**Hair Score:** {results2['hair']:.1f}/100")
                
                # Show winner
                st.divider()
                st.subheader(f"🏆 Winner: Person {winner}!")
                st.image(winner_pil, use_column_width=True)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
            finally:
                # Cleanup temp files
                os.unlink(temp_file1.name)
                os.unlink(temp_file2.name)
    else:
        st.info("👆 Upload two images to compare")

if __name__ == "__main__":
    main()
