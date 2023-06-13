import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(image, kernel)

def pencil_sketch_grey(image):
    gray_image = greyscale(image)
    inverted_image = cv2.bitwise_not(gray_image)
    blurred_image = cv2.GaussianBlur(inverted_image, (111, 111), 0)
    inverted_blurred_image = cv2.bitwise_not(blurred_image)
    return cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

def invert(image):
    return cv2.bitwise_not(image)

def summer(image):
    summer_filter = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
    return cv2.transform(image, summer_filter)

def winter(image):
    winter_filter = np.array([[0.7, 0.3, 0.3],
                              [0.3, 0.7, 0.3],
                              [0.3, 0.3, 0.7]])
    return cv2.transform(image, winter_filter)

def cartoon_effect(image):
    gray = greyscale(image)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def blur_image(image, rate):
    ksize = int(rate * 5)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def brighten_image(image, amount):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - amount
    v = np.where(v <= lim, v + amount, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def enhance_details(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def image_comparison(img1, img2):
    col1, col2 = st.beta_columns(2)
    col1.subheader("Original Image")
    col1.image(img1, use_column_width=True)
    col2.subheader("Processed Image")
    col2.image(img2, use_column_width=True)

def main_loop():
    st.title("Image Filters")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        original_image = Image.open(uploaded_image)
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)

        selected_filter = st.selectbox("Select a filter", ["None", "Gray Effect", "Sepia Effect", "Pencil Sketch",
                                                           "Invert Effect", "Summer", "Winter", "Cartoon Effect"])
        blur_rate = st.slider("Blur Rate", 0.0, 1.0, 0.5)
        brightness_amount = st.slider("Brightness", -100, 100, 0)

        apply_enhancement_filter = st.checkbox("Apply Enhancement Filter")
        apply_rgb_adjustments = st.checkbox("Apply RGB Adjustments")
        apply_saturation_adjustment = st.checkbox("Apply Saturation Adjustment")
        apply_shadow_adjustment = st.checkbox("Apply Shadow Adjustment")

        rgb_sliders = {
            "Red": st.slider("Red", -255, 255, 0, key="red"),
            "Green": st.slider("Green", -255, 255, 0, key="green"),
            "Blue": st.slider("Blue", -255, 255, 0, key="blue")
        }

        saturation = st.slider("Saturation", 0.0, 2.0, 1.0)
        shadow = st.slider("Shadow", 0.0, 1.0, 0.5)

        if selected_filter != "None":
            processed_image = np.array(original_image.convert("RGB"))
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

            if apply_enhancement_filter:
                processed_image = enhance_details(processed_image)

            if apply_rgb_adjustments:
                processed_image[..., 0] = np.clip(processed_image[..., 0] + rgb_sliders["Red"], 0, 255)
                processed_image[..., 1] = np.clip(processed_image[..., 1] + rgb_sliders["Green"], 0, 255)
                processed_image[..., 2] = np.clip(processed_image[..., 2] + rgb_sliders["Blue"], 0, 255)

            if apply_saturation_adjustment:
                hsv_img = cv2.cvtColor(np.copy(processed_image), cv2.COLOR_BGR2HSV)
                hsv_img[..., 1] = np.clip(hsv_img[..., 1] * saturation, 0, 255)
                processed_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

            if apply_shadow_adjustment:
                processed_image = cv2.addWeighted(processed_image, shadow, processed_image, 0, 0)

            image_comparison(original_image, Image.fromarray(processed_image))

        else:
            st.warning("Please select a filter.")
    else:
        st.warning("Please upload an image.")

if __name__ == "__main__":
    main_loop()

