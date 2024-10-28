import streamlit as st
import pandas as pd
import numpy as np
from itables.streamlit import interactive_table
import pyarrow
from streamlit.components.v1 import html
from streamlit.components.v1.components import MarshallComponentException
from PIL import Image
from streamlit_navigation_bar import st_navbar
import pages as pg
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from vertexai.language_models import TextEmbeddingModel

import os
from google.cloud import storage
from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import aiplatform
import base64
from google.cloud import storage
from google.protobuf import struct_pb2
import typing

import google.cloud.aiplatform as aiplatform
import google.cloud.aiplatform_v1beta1 as aiplatform_v1beta1
PROJECT_ID = os.environ.get("bagchi-genai-bb")
LOCATION = os.environ.get("us_central1")
BUCKET = "bagchi-genai-bb"
BUCKET_URI = f"gs://{BUCKET}/"

vertexai.init(project=PROJECT_ID, location=LOCATION)
print(f"Using vertexai version: {vertexai.__version__}")


class EmbeddingResponse(typing.NamedTuple):
  text_embedding: typing.Sequence[float]
  image_embedding: typing.Sequence[float]


class EmbeddingPredictionClient:
  """Wrapper around Prediction Service Client."""
  def __init__(self, project : str,
    location : str = "us-central1",
    api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
    client_options = {"api_endpoint": api_regional_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)  
    self.location = location
    self.project = "bagchi-genai-bb"

  def get_embedding(self, text : str = None, image_bytes : bytes = None):
    if not text and not image_bytes:
      raise ValueError('At least one of text or image_bytes must be specified.')

    instance = struct_pb2.Struct()
    if text:
      instance.fields['text'].string_value = text

    if image_bytes:
      encoded_content = base64.b64encode(image_bytes).decode("utf-8")
      image_struct = instance.fields['image'].struct_value
      image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

    instances = [instance]
    endpoint = (f"projects/{self.project}/locations/{self.location}"
      "/publishers/google/models/multimodalembedding@001")
    print(f"Calling Vertex AI  : {endpoint}")

    response = self.client.predict(endpoint=endpoint, instances=instances)
    print(f" Vertex AI  Response : {response}")

    text_embedding = None
    if text:    
      text_emb_value = response.predictions[0]['textEmbedding']
      text_embedding = [v for v in text_emb_value]

    image_embedding = None
    if image_bytes:    
      image_emb_value = response.predictions[0]['imageEmbedding']
      image_embedding = [v for v in image_emb_value]

    return EmbeddingResponse(
      text_embedding=text_embedding,
      image_embedding=image_embedding)

print(f"Calling EmbeddingPredictionClient with project {PROJECT_ID}")

client = EmbeddingPredictionClient(project=PROJECT_ID)

print(f"EmbeddingPredictionClient returned : {client}")

# storage_client = storage.Client()
# bucket = storage_client.get_bucket(BUCKET)

@st.cache_resource
def load_models() -> tuple[GenerativeModel, GenerativeModel]:
    """Load Gemini 1.5 Flash and Pro models."""
    return GenerativeModel("gemini-1.5-flash"), GenerativeModel("gemini-1.5-pro")

gemini_15_flash, gemini_15_pro = load_models()
print(f"Models loaded: {gemini_15_flash}")

def get_gemini_response(
    model: GenerativeModel,
    contents: list,
    generation_config: GenerationConfig = GenerationConfig(
        temperature=0.1, max_output_tokens=2048
    ),
    stream: bool = True,
) -> str:
    """Generate a response from the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )
    print(f"Response: {responses}")

    if not stream:
        return responses.text

    final_response = []
    for r in responses:
        try:
            final_response.append(r.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)


def get_model_name(model: GenerativeModel) -> str:
    """Get Gemini Model Name"""
    model_name = model._model_name.replace(  # pylint: disable=protected-access
        "publishers/google/models/", ""
    )
    return f"`{model_name}`"


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


def generate_story(text_gen_prompt: str) -> str:
    print(f"calling gemini response: {gemini_15_flash}")

    temperature = 0.30
    max_output_tokens = 2048
    
    config = GenerationConfig(
    temperature=temperature, max_output_tokens=max_output_tokens
    )

    response =  get_gemini_response(
    gemini_15_flash,  # Use the selected model
    text_gen_prompt,
    generation_config=config,
    )
    print(f"received gemini response: {response}")

    return response

def generate_image_classification(images_content) -> str:
    print(f"calling gemini response: {gemini_15_flash}")

    temperature = 0.30
    max_output_tokens = 2048
    
    config = GenerationConfig(
    temperature=temperature, max_output_tokens=max_output_tokens
    )

    response =  get_gemini_response(
    gemini_15_flash,  # Use the selected model
    images_content,
    generation_config=config,
    )
    print(f"received gemini response: {response}")

    return response


def search(image_bytes,text_search):
  model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
  text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")

  # embedding_dimension = 128
  #  converted_query_to_embedding = client.get_embedding(image_bytes=image_bytes)
  # Set variables for the current deployed index.
  API_ENDPOINT="875183697.us-central1-104454103637.vdb.vertexai.goog"
  INDEX_ENDPOINT="projects/104454103637/locations/us-central1/indexEndpoints/340703469075693568"
  DEPLOYED_INDEX_ID="games_demo_deployed_index_1730123541168"

  image = Image.load_from_file(
    "gs://bagchi-genai-bb/input_images/car-front.png"
    )
  embeddings = model.get_embeddings(
    image=image,
    contextual_text=text_search,
  )
  print(f"Image Embedding: {embeddings}")
   # Configure Matching Engine Index Client
  client_options = {
        "api_endpoint": API_ENDPOINT
    }
  my_index_endpoint_id = "340703469075693568"  # @param {type:"string"}
  my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(my_index_endpoint_id)
  print(f"my_index_endpoint: {my_index_endpoint}")


  # text_embeddings = text_embedding_model.get_embeddings([text_search])
  # vector = text_embeddings[0].values
  # print(f"Text Vector: {text_embeddings[0].values}")

  # run query
  print(f"run query: my_index_endpoint")

  response = my_index_endpoint.find_neighbors(
      deployed_index_id=DEPLOYED_INDEX_ID, queries=[embeddings], num_neighbors=10
  )
  print(f"Query Response {response}")

