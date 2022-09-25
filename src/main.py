import streamlit as st
import pandas as pd 
import numpy as np
import ffmpeg


st.title('Find the fight 👊🏼💥')
st.subheader('Raw data')

@st.cache
def load_video(filename):
    # stream = ffmpeg.input(filename)
    
    return filename.getvalue()


filename = st.file_uploader(help="Select a input video...",
                            label="Input video for fight detection")

# subpath = "noFight" if "nofi" in filename.name else "fight"
# video_path = f"fight-detection-surv-dataset/{subpath}/{filename.name}"

data_load_state = st.text('Loading data...')
video_bytes = load_video(filename=filename)
data_load_state.text("Done! (using st.cache)")
st.video(video_bytes)
