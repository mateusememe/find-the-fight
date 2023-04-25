import streamlit as st
import pandas as pd 
import numpy as np

st.title('Find the fight 👊🏼💥')

@st.cache_resource
def load_video(filename):
    # stream = ffmpeg.input(filename)
    if filename != None:
      return filename.getvalue()

# Mocked data as a list of dictionaries with explicit column names and data types
_data = [{'Name': 'John', 'Age': 25, 'City': 'New York'},
        {'Name': 'Alice', 'Age': 30, 'City': 'San Francisco'},
        {'Name': 'Bob', 'Age': 27, 'City': 'Chicago'}]

# Create a mocked DataFrame from the list of dictionaries
data = pd.DataFrame(_data)

# Sidebar options
sidebar_options = ["Histogram", "Scatterplot"]
sidebar_selection = st.sidebar.radio("Select plot type:", sidebar_options)

# Data filtering options
filter_col = st.sidebar.selectbox("Select column to filter:", data.columns)
filter_val = st.sidebar.text_input("Filter value:")

# Filter data based on user input
filtered_data = data[data[filter_col].str.contains(filter_val)]


filename = st.file_uploader(help="Select a input video...",
                            label="Input video for fight detection")

# subpath = "noFight" if "nofi" in filename.name else "fight"
# video_path = f"fight-detection-surv-dataset/{subpath}/{filename.name}"

data_load_state = st.text('Loading data...')
video_bytes = load_video(filename=filename)
data_load_state.text("Done! (using st.cache)")

st.video(video_bytes)
