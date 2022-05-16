import streamlit as st
import pandas as pd 
import numpy as np
import ffmpeg
st.title('Find the fight 👊🏼💥')
st.subheader('Raw data')

@st.cache
def load_video(filename: str):
    stream = ffmpeg.input(filename)
    stream = ffmpeg.output(filename.replace(".avi", ".mp4"))
    st.text("aaaa",type(stream))
    video_file = open(filename, 'r')
    video_bytes = video_file.read()
    return video_bytes


filename = st.file_uploader(help="Select a input video...",
                            label="Input video for fight detection",
                            type="avi, mpeg, mp4")

data_load_state = st.text('Loading data...')
video_bytes = load_video(filename="datasets/UT-Iteraction/segmented_set1/0_1_4.avi")
data_load_state.text("Done! (using st.cache)")
st.video(video_bytes)