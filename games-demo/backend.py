import streamlit as st
from PIL import Image
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from vertexai.language_models import TextEmbeddingModel

from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import aiplatform
import base64
from google.protobuf import struct_pb2
import typing
import google.cloud.aiplatform as aiplatform

from visionai.python.gapic.visionai import visionai_v1
from visionai.python.net import channel

PROJECT_ID = "bagchi-genai-bb"
LOCATION = "us-central1"
BUCKET = "bagchi-genai-bb"
BUCKET_URI = f"gs://{BUCKET}/"


@st.cache_resource
def get_vertexai_session():
    """Getting handle to vertex ai"""
    print(f"vertexai.init called : {PROJECT_ID}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print(f"Using vertexai version: {vertexai.__version__}")


get_vertexai_session()


class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]


class EmbeddingPredictionClient:
    """Wrapper around Prediction Service Client."""

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com",
    ):
        client_options = {"api_endpoint": api_regional_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options
        )
        self.location = LOCATION
        self.project = PROJECT_ID

    def get_embedding(self, text: str = None, image_bytes: bytes = None):
        if not text and not image_bytes:
            raise ValueError("At least one of text or image_bytes must be specified.")

        instance = struct_pb2.Struct()
        if text:
            instance.fields["text"].string_value = text

        if image_bytes:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            image_struct = instance.fields["image"].struct_value
            image_struct.fields["bytesBase64Encoded"].string_value = encoded_content

        instances = [instance]
        endpoint = (
            f"projects/{self.project}/locations/{self.location}"
            "/publishers/google/models/multimodalembedding@001"
        )
        print(f"Calling Vertex AI  : {endpoint}")

        response = self.client.predict(endpoint=endpoint, instances=instances)
        print(f" Vertex AI  Response : {response}")

        text_embedding = None
        if text:
            text_emb_value = response.predictions[0]["textEmbedding"]
            text_embedding = [v for v in text_emb_value]

        image_embedding = None
        if image_bytes:
            image_emb_value = response.predictions[0]["imageEmbedding"]
            image_embedding = [v for v in image_emb_value]

        return EmbeddingResponse(
            text_embedding=text_embedding, image_embedding=image_embedding
        )


print(f"Calling EmbeddingPredictionClient with project {PROJECT_ID}")
client = EmbeddingPredictionClient(project=PROJECT_ID)
print(f"EmbeddingPredictionClient returned : {client}")


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
        temperature=0.1, max_output_tokens=200
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
    max_output_tokens = 100

    config = GenerationConfig(
        temperature=temperature, max_output_tokens=max_output_tokens
    )

    response = get_gemini_response(
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

    response = get_gemini_response(
        gemini_15_flash,  # Use the selected model
        images_content,
        generation_config=config,
    )
    print(f"received gemini response: {response}")

    return response

@st.cache_resource
def get_warehouse_client():
    warehouse_endpoint = channel.get_warehouse_service_endpoint(
        channel.Environment["PROD"]
    )
    warehouse_client = visionai_v1.WarehouseClient(
        client_options={"api_endpoint": warehouse_endpoint}
    )
    return warehouse_client


# @st.cache_resource
# def get_storage_bucket():
#   print(f"storage_client called : {PROJECT_ID}")
#   storage_client = storage.Client(project=PROJECT_ID)
#   bucket = storage_client.get_bucket(BUCKET)
#   print(f"storage_client returned : {bucket}")
#   return bucket

# get_storage_bucket()


def search_image_warehouse(uploaded_file: bytes, text_query: str,search_by_image : bool) -> list:
    wh_client = get_warehouse_client()
    MAX_RESULTS = 10  # @param {type: "integer"} Set to 0 to allow all results.
    QUERY = text_query  # @param {type: "string"}
    endpoint_name = "projects/104454103637/locations/us-central1/indexEndpoints/games-search-endpoint-demo"
    print(f"calling endpoint {endpoint_name}")
    print(f"Image Query {uploaded_file}")
    if search_by_image == False:
        results = wh_client.search_index_endpoint(
            visionai_v1.SearchIndexEndpointRequest(
                index_endpoint=endpoint_name,
                text_query=QUERY,
            ),
        )
    else:
        print(f"Calling Image Query {uploaded_file}")
        with open("search_image", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with open("search_image", "rb") as localfile:
            image_content = localfile.read()
        print(f"Calling Image Endpoint {image_content}")

        results = wh_client.search_index_endpoint(
            visionai_v1.SearchIndexEndpointRequest(
                index_endpoint=endpoint_name,
                image_query=visionai_v1.ImageQuery(
                    input_image=image_content,
                ),
            ),
        )
        # print(f"Called Image Query {results}")

    print(f"received respnse {results}")

    results_cnt = 0
    asset_names = []
    asset_relevances = []
    for r in results:
        if float(r.relevance) > 0.4:
            asset_names.append(r.asset)
            asset_relevances.append(r.relevance)
            results_cnt += 1
        if results_cnt >= MAX_RESULTS:
            break

    # Sort asset_names based on asset_relevances in descending order
    sorted_data = sorted(zip(asset_relevances, asset_names), reverse=True)
    asset_names = [asset for _, asset in sorted_data]
    # print(f"sorted respnse {asset_names}")

    uris = list(
        map(
            lambda asset_name: wh_client.generate_retrieval_url(
                visionai_v1.GenerateRetrievalUrlRequest(
                    name=asset_name,
                )
            ).signed_uri,
            asset_names,
        )
    )
    print(f"Images URIS size {len(uris)}")

    return uris
