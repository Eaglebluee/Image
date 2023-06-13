import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison
import io

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright

def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img

def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon_img = cv2.bitwise_and(color, color, mask=edges)
    return cartoon_img

def greyscale(img):
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return greyscale

def sepia(img):
    img_sepia = np.array(img, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))
    img_sepia[np.where(img_sepia > 255)] = 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia

def pencil_sketch_grey(img):
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_gray

def invert(img):
    inv = cv2.bitwise_not(img)
    return inv

def summer(img):
    summer_img = img.copy()
    summer_img[..., 0] = np.clip(img[..., 0] * 1.2, 0, 255)
    summer_img[..., 2] = np.clip(img[..., 2] * 0.8, 0, 255)
    return summer_img

def winter(img):
    winter_img = img.copy()
    winter_img[..., 0] = np.clip(img[..., 0] * 0.8, 0, 255)
    winter_img[..., 2] = np.clip(img[..., 2] * 1.2, 0, 255)
    return winter_img

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def apply_face_filter(image, filter_type):
    # Placeholder function for applying face filters
    # Replace this function with the actual implementation of your face filters
    # You can apply different filters based on the filter_type parameter
    return image

def save_image(image, filename):
    image_pil = Image.fromarray(image)
    image_pil.save(filename)

def main_loop():
    st.title("Image Editor")
    st.subheader("You can edit and apply Filters to your images!")

    filters = {
        "Original Image": "Original Image without any modifications",
        "Cartoon Effect": "Apply a cartoon effect to the image",
        "Gray Effect": "Convert the image to grayscale",
        "Sepia Effect": "Apply a sepia tone effect to the image",
        "Pencil Sketch": "Create a pencil sketch effect",
        "Invert Effect": "Invert the colors of the image",
        "Summer": "Apply a summer color effect",
        "Winter": "Apply a winter color effect"
    }

    face_filters = {
        "Face Filter 1": "Placeholder for Face Filter 1",
        "Face Filter 2": "Placeholder for Face Filter 2",
        "Face Filter 3": "Placeholder for Face Filter 3",
        "Face Filter 4": "Placeholder for Face Filter 4"
    }

    selected_filter = st.sidebar.selectbox("Filters", list(filters.keys()), format_func=lambda x: x)
    filter_tooltip = filters[selected_filter]
    st.sidebar.text(filter_tooltip)

    detect_faces_toggle = st.sidebar.checkbox("Detect Faces")

    if detect_faces_toggle:
        face_filter_type = st.sidebar.selectbox("Face Filters", list(face_filters.keys()), format_func=lambda x: x)
        face_filter_tooltip = face_filters[face_filter_type]
        st.sidebar.text(face_filter_tooltip)

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    if selected_filter == "Cartoon Effect":
        processed_image = cartoon_effect(original_image)
    elif selected_filter == "Gray Effect":
        processed_image = greyscale(original_image)
    elif selected_filter == "Sepia Effect":
        processed_image = sepia(original_image)
    elif selected_filter == "Pencil Sketch":
        processed_image = pencil_sketch_grey(original_image)
    elif selected_filter == "Invert Effect":
        processed_image = invert(original_image)
    elif selected_filter == "Summer":
        processed_image = summer(original_image)
    elif selected_filter == "Winter":
        processed_image = winter(original_image)
    else:
        processed_image = np.copy(original_image)

    processed_image = blur_image(processed_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    if detect_faces_toggle:
        faces = detect_faces(processed_image)
        for (x, y, w, h) in faces:
            face_roi = processed_image[y:y+h, x:x+w]
            face_roi = apply_face_filter(face_roi, face_filter_type)
            processed_image[y:y+h, x:x+w] = face_roi

    st.text("Original Image vs Processed Image")

    image_comparison(
        img1=original_image,
        img2=processed_image,
    )

    # Download link for the processed image
    processed_image_pil = Image.fromarray(processed_image)
    output = io.BytesIO()
    processed_image_pil.save(output, format='JPEG')
    output.seek(0)
    st.download_button("Download Processed Image", data=output, file_name="processed_image.jpg")

if __name__ == '__main__':
    main_loop()
