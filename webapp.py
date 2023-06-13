import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def main_loop():
    st.title("Image Filters")

    filters = {
        "Cartoon Effect": "Apply a cartoon effect to the image",
        "Gray Effect": "Convert the image to grayscale",
        "Sepia Effect": "Apply a sepia tone to the image",
        "Pencil Sketch": "Create a pencil sketch effect",
        "Invert Effect": "Invert the colors of the image",
        "Summer": "Apply a summer color effect",
        "Winter": "Apply a winter color effect",
        "Detect Faces": "Detect and highlight faces in the image"
    }

    selected_filter = st.sidebar.selectbox("Filters", list(filters.keys()), format_func=lambda x: x)
    filter_tooltip = filters[selected_filter]
    st.sidebar.text(filter_tooltip)

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)

    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)

    detect_faces = selected_filter == "Detect Faces"

    if detect_faces:
        face_filters = {
            "Default": "Remove any applied face filter",
            "Glasses 1": "Apply Glasses 1 filter to the detected eyes",
            "Glasses 2": "Apply Glasses 2 filter to the detected eyes",
            "Glasses 3": "Apply Glasses 3 filter to the detected eyes",
            "Glasses 4": "Apply Glasses 4 filter to the detected eyes",
            "Glasses 5": "Apply Glasses 5 filter to the detected eyes"
        }
        selected_face_filter = st.sidebar.selectbox("Face Filters", list(face_filters.keys()), format_func=lambda x: x)
        face_filter_tooltip = face_filters[selected_face_filter]
        st.sidebar.text(face_filter_tooltip)

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    processed_image = np.copy(original_image)

    if selected_filter == "Cartoon Effect":
        processed_image = cartoon_effect(processed_image)
    elif selected_filter == "Gray Effect":
        processed_image = greyscale(processed_image)
    elif selected_filter == "Sepia Effect":
        processed_image = sepia(processed_image)
    elif selected_filter == "Pencil Sketch":
        processed_image = pencil_sketch_grey(processed_image)
    elif selected_filter == "Invert Effect":
        processed_image = invert(processed_image)
    elif selected_filter == "Summer":
        processed_image = summer(processed_image)
    elif selected_filter == "Winter":
        processed_image = winter(processed_image)

    processed_image = blur_image(processed_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)

    if detect_faces:
        processed_image, faces, eyes = detect_and_draw_faces(processed_image)
        if selected_face_filter != "Default":
            processed_image = apply_face_filter(processed_image, eyes, selected_face_filter)

    st.image(processed_image, use_column_width=True)


def cartoon_effect(image, num_down=2, num_bilateral=7):
    img_color = image
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)

    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)

    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(img_color, img_edge)


def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def sepia(image):
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_matrix)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image


def pencil_sketch_grey(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21, 21), sigmaX=0, sigmaY=0)
    img_blend = cv2.divide(img_gray, img_blur, scale=256.0)
    return img_blend


def invert(image):
    return cv2.bitwise_not(image)


def summer(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_hsv[:, :, 1] = image_hsv[:, :, 1] + 25
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def winter(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_hsv[:, :, 1] = image_hsv[:, :, 1] - 25
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def blur_image(image, rate):
    ksize = int(rate * 5)
    return cv2.blur(image, (ksize, ksize))


def brighten_image(image, amount):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = np.where((hsv_image[:, :, 2] + amount) > 255, 255, hsv_image[:, :, 2] + amount)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def detect_and_draw_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    return image, faces, eyes


def apply_face_filter(image, eyes, selected_face_filter):
    glasses_filter_path = f"path/to/{selected_face_filter}.png"
    glasses_image = cv2.imread(glasses_filter_path, cv2.IMREAD_UNCHANGED)

    for (ex, ey, ew, eh) in eyes:
        glasses_width = int(ew * 1.5)
        glasses_height = int(eh * 0.6)
        glasses_image = cv2.resize(glasses_image, (glasses_width, glasses_height))

        x_offset = ex - int(0.2 * ew)
        y_offset = ey + int(0.4 * eh)

        alpha_s = glasses_image[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            image[y_offset:y_offset + glasses_height, x_offset:x_offset + glasses_width, c] = \
                (alpha_s * glasses_image[:, :, c] + alpha_l * image[y_offset:y_offset + glasses_height,
                                                                    x_offset:x_offset + glasses_width, c])

    return image


if __name__ == "__main__":
    main_loop()
