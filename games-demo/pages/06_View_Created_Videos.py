import streamlit as st
from google.cloud import storage
import datetime
# Streamlit configuration
st.set_page_config(page_title="View Vides", layout="wide")

# --- GCP Configuration ---
# # Replace these with your actual GCP project and bucket details
PROJECT_ID = "veo-testing"
BUCKET_NAME = "bagchi-genai-bb-veo-testing"

# Initialize Google Cloud Storage client
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# --- Helper Functions ---

def download_video_as_bytes(blob_name):
  """Downloads the video blob as bytes."""

  blob = bucket.blob(blob_name)
  return blob.download_as_bytes()

# --- Streamlit App ---

st.title("Google Cloud Storage Video Player")

# Get a list of video files in the bucket
blobs = list(bucket.list_blobs(prefix="veo-output/")) # Assuming videos are stored in a 'videos/' folder
video_files = [blob.name for blob in blobs if blob.name.endswith(('.mp4', '.mov', '.avi'))]

# Select box to choose a video
selected_video = st.selectbox("Select a video:", video_files)

if selected_video:
  try:
    # Download the video as bytes
    video_bytes = download_video_as_bytes(selected_video)

    # Display the video
    st.video(video_bytes)

  except Exception as e:
    st.error(f"Error loading video: {e}")