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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    hdr = cv2.detailEnhance(img_rgb, sigma_s=12, sigma_r=0.15)
    hdr_bgr = cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)  # Convert back to BGR format
    return hdr_bgr


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
    img_sepia = np.array(img, dtype=np.float64)  # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))  # multiplying image with sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
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


def detect_faces(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return img


def main_loop():

    st.sidebar.title("Filter Options")

    filters = {
        "Original Image": "Display the original image",
        "Detect Faces": "Detect and draw boxes around faces and eyes",
        "Cartoon Effect": "Apply a cartoon effect",
        "Gray Effect": "Convert the image to grayscale",
        "Sepia Effect": "Apply a sepia effect",
        "Pencil Sketch": "Convert the image to a pencil sketch",
        "Invert Effect": "Invert the colors of the image",
        "Summer": "Apply a summer color effect",
        "Winter": "Apply a winter color effect"
    }

    selected_filter = st.sidebar.selectbox("Filters", list(filters.keys()), format_func=lambda x: x)
    filter_tooltip = filters[selected_filter]
    st.sidebar.text(filter_tooltip)

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)

    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    rgb_sliders = {
        "Red": st.sidebar.slider("Red", min_value=0, max_value=255, value=0),
        "Green": st.sidebar.slider("Green", min_value=0, max_value=255, value=0),
        "Blue": st.sidebar.slider("Blue", min_value=0, max_value=255, value=0)
    }
    saturation = st.sidebar.slider("Saturation", min_value=0.0, max_value=2.0, value=1.0)
    shadow = st.sidebar.slider("Shadow", min_value=0.0, max_value=1.0, value=0.5)
    
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    hdr = cv2.detailEnhance(img_rgb, sigma_s=12, sigma_r=0.15)
    hdr_bgr = cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)  # Convert back to BGR format
    return hdr_bgr


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
    img_sepia = np.array(img, dtype=np.float64)  # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))  # multiplying image with sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
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


def detect_faces(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return img


def main_loop():

    st.sidebar.title("Filter Options")

    filters = {
        "Original Image": "Display the original image",
        "Detect Faces": "Detect and draw boxes around faces and eyes",
        "Cartoon Effect": "Apply a cartoon effect",
        "Gray Effect": "Convert the image to grayscale",
        "Sepia Effect": "Apply a sepia effect",
        "Pencil Sketch": "Convert the image to a pencil sketch",
        "Invert Effect": "Invert the colors of the image",
        "Summer": "Apply a summer color effect",
        "Winter": "Apply a winter color effect"
    }

    selected_filter = st.sidebar.selectbox("Filters", list(filters.keys()), format_func=lambda x: x)
    filter_tooltip = filters[selected_filter]
    st.sidebar.text(filter_tooltip)

    rgb_sliders = {
        "Red": 0,
        "Green": 0,
        "Blue": 0
    }

    saturation = 1.0
    shadow = 0.0

    apply_rgb_adjustments = st.sidebar.checkbox("Apply RGB Adjustments")
    if apply_rgb_adjustments:
        rgb_sliders["Red"] = st.sidebar.slider("Red", min_value=-255, max_value=255, value=0)
        rgb_sliders["Green"] = st.sidebar.slider("Green", min_value=-255, max_value=255, value=0)
        rgb_sliders["Blue"] = st.sidebar.slider("Blue", min_value=-255, max_value=255, value=0)

    apply_saturation_adjustment = st.sidebar.checkbox("Apply Saturation Adjustment")
    if apply_saturation_adjustment:
        saturation = st.sidebar.slider("Saturation", min_value=0.0, max_value=2.0, value=1.0)

    apply_shadow_adjustment = st.sidebar.checkbox("Apply Shadow Adjustment")
    if apply_shadow_adjustment:
        shadow = st.sidebar.slider("Shadow", min_value=0.0, max_value=1.0, value=0.0)

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)

    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    processed_image = np.copy(original_image)

    if selected_filter == "Detect Faces":
        processed_image = detect_faces(processed_image)
    elif selected_filter == "Cartoon Effect":
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

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    if apply_rgb_adjustments:
        processed_image[..., 0] = np.clip(processed_image[..., 0] + rgb_sliders["Red"], 0, 255)
        processed_image[..., 1] = np.clip(processed_image[..., 1] + rgb_sliders["Green"], 0, 255)
        processed_image[..., 2] = np.clip(processed_image[..., 2] + rgb_sliders["Blue"], 0, 255)

    if apply_saturation_adjustment:
        hsv_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        hsv_img[..., 1] = np.clip(hsv_img[..., 1] * saturation, 0, 255)
        processed_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    if apply_shadow_adjustment:
        processed_image = cv2.addWeighted(processed_image, 1.0, np.zeros_like(processed_image), 1.0 - shadow, 0)

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


if __name__ == "__main__":
    main_loop()
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
    processed_image[..., 1] = np.clip(processed_image[..., 1] * saturation, 0, 255)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR)

    # Adjust shadows
    processed_image = processed_image.astype('float32')
    processed_image[..., 2] = np.clip(processed_image[..., 2] * (1 - shadow), 0, 255)
    processed_image = processed_image.astype('uint8')

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

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


if __name__ == "__main__":
    main_loop()
