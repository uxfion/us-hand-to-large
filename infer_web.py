import streamlit as st
import random
import os
import io
import datetime
from PIL import Image, ImageFilter
import SimpleITK as sitk
import requests


def random_image_from_folder(folder_path):
    """ 随机从指定文件夹中选择一张图片 """
    files = os.listdir(folder_path)
    random_file = random.choice(files)
    image_path = os.path.join(folder_path, random_file)
    return image_path


# Set page config to make the layout use the full page width
st.set_page_config(layout="wide")

# Title of the webpage
st.title('Medical Image Quality Improvement')

col1, col2 = st.columns(2)

# File uploader allows the user to add their own image
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Dropdown for model selection
with col2:
    model_select = st.selectbox('Model select', ['Super Resolution Model'])

parameter_col, button_col = st.columns(2)
with parameter_col:
    if model_select == 'Super Resolution Model':
        # parameter = st.slider('Select parameter', 1, 19, step=1, value=8)
        parameter = -1
with button_col:
    _, demo_col, infer_col, _ = st.columns(4)
    with demo_col:
        demo_button = st.button('Demo')
    with infer_col:
        infer_button = st.button('Infer')

if demo_button:
    demo_file = random_image_from_folder('./datasets/all/test')  # 随机选择图片
    st.session_state['demo_image'] = demo_file
    st.write(f"Pick a random image: {demo_file}")
    st.image(demo_file, caption='Random image', width=250)


# Button to perform inference
if infer_button:
    uploaded_file = st.session_state.get('demo_image', uploaded_file)
    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            with open(uploaded_file, "rb") as f:
                contents = f.read()
        else:
            contents = uploaded_file.getvalue()

        # use for web api
        files = {"file": contents}
        # params = {"para": parameter}

        # use for PIL Image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Use Streamlit's columns feature to display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Image before infer', use_column_width=True)

        # Improve image quality
        # improved_image = improve_image_quality(image, model, parameter)
        # print(f"{datetime.datetime.now()} 推理中...")
        # print("图像信息")
        # print(f"  - 尺寸: {image.size}")
        # print(f"  - 格式: {image.format}")
        # print(f"  - exif: {image.getexif()}")

        # print(f"推理模型: {model_select}")

        if model_select == 'Super Resolution Model':
            response = requests.post("http://127.0.0.1:1234/process/", files=files)
            if response.status_code == 200:
                improved_image = Image.open(io.BytesIO(response.content))
            else:
                st.error("Error processing image.")

        print(f"{datetime.datetime.now()} 推理完成\n\n\n")

        # Display the image after improvement
        with col2:
            st.image(improved_image, caption='Image after infer', use_column_width=True)

        # Download button
        buf = io.BytesIO()
        improved_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download image",
                           data=byte_im,
                           file_name="improved_image.png",
                           mime="image/png")
    else:
        st.error("Please upload an image to infer.")
