import streamlit as st

def main():
    st.title("Video Upload")

    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov'])

    if uploaded_file is not None:
        # Process the uploaded video here
        # You can use libraries like OpenCV or FFmpeg to work with videos
        # Display the video using st.video() or perform other operations

        st.video(uploaded_file)

if __name__ == '__main__':
    main()
