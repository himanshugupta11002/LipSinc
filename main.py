import streamlit as st
import os
import tensorflow as tf
import ffmpeg
from utils import load_data, num_to_char
from modelutil import load_model

# Set page configuration
st.set_page_config(layout='wide')

# Sidebar setup
with st.sidebar:
    st.image('https://images.unsplash.com/photo-1500622944204-b135684e99fd?q=80&w=2061&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D', use_column_width=True)
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

# Main title
st.title('LipNet Full Stack App')

# Select video from options
options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox('Choose a video', options)

# Generate two columns layout
col1, col2 = st.columns(2)

def convert_video(input_path, output_path='test_video.mp4'):
    (
        ffmpeg
        .input(input_path)
        .output(output_path, vcodec='libx264')
        .run(overwrite_output=True)
    )
    return output_path

@st.cache
def load_and_process_video(video_path):
    video, _ = load_data(video_path)
    return tf.convert_to_tensor(video)

if selected_video:
    # Display video in col1
    with col1:
        st.info('Video display')
        video_path = os.path.join('data', 's1', selected_video)
        convert_video(video_path)  # Convert video to a suitable format
        # Rendering inside of the app
        video_file = open('test_video.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    # Load and process video, predict, and show results in col2
    with col2:
        st.info('Machine Learning Model Prediction')

        # Load and preprocess video
        video = load_and_process_video(video_path)

        # Load model and predict
        model = load_model()
        preds = model.predict(tf.expand_dims(video, axis=0))

        # Decode predictions
        decoder = tf.keras.backend.ctc_decode(preds, input_length=[75], greedy=True)[0][0]
        prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

        st.text('Predicted Text:')
        st.write(prediction)
