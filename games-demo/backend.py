import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)

from google.cloud import aiplatform
import base64
from google.protobuf import struct_pb2
import typing
import google.cloud.aiplatform as aiplatform
from google.auth import default, transport
import openai
import time
from visionai.python.gapic.visionai import visionai_v1
from visionai.python.net import channel
from google.cloud import storage

PROJECT_ID = "bagchi-genai-bb"
LOCATION = "us-central1"
BUCKET = "bagchi-genai-bb"
BUCKET_URI = f"gs://{BUCKET}/"
IMAGE_WAREHOUSE_ENDPOINT_NAME = "projects/104454103637/locations/us-central1/indexEndpoints/games-search-endpoint-demo"
VEO_BUCKET = "bagchi-genai-bb-veo-testing"
VEO_OUTPUT_FOLDER = "veo-output"
VEO_PROJECT_ID = "veo-testing"
VEO_IMAGES_FOLDER = "veo-input-images"


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


# print(f"Calling EmbeddingPredictionClient with project {PROJECT_ID}")
# # client = EmbeddingPredictionClient(project=PROJECT_ID)
# print(f"EmbeddingPredictionClient returned : {client}")


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


def generate_story(text_gen_prompt: str, response_schema) -> str:
    print(f"calling gemini response: {gemini_15_flash}")

    temperature = 0.30
    max_output_tokens = 2048
    if response_schema is not None:
        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_schema=response_schema,
            response_mime_type="application/json",
        )
    else:
        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    response = get_gemini_response(
        gemini_15_flash,  # Use the selected model
        text_gen_prompt,
        generation_config=config,
    )
    print(f"received gemini response: {response}")

    return response


def generate_image_classification(images_content) -> str:
    # print(f"calling gemini response: {gemini_15_flash}")

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
    # print(f"received gemini response: {response}")

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


wh_client = get_warehouse_client()

# @st.cache_resource
# def get_storage_bucket():
#   print(f"storage_client called : {PROJECT_ID}")
#   storage_client = storage.Client(project=PROJECT_ID)
#   bucket = storage_client.get_bucket(BUCKET)
#   print(f"storage_client returned : {bucket}")
#   return bucket

# get_storage_bucket()


def search_image_warehouse(
    uploaded_file: bytes, text_query: str, search_by_image: bool
) -> list:
    MAX_RESULTS = 10  # @param {type: "integer"} Set to 0 to allow all results.
    QUERY = text_query  # @param {type: "string"}
    # print(f"calling endpoint {endpoint_name}")
    # print(f"Image Query {uploaded_file}")
    if search_by_image == False:
        print(f"Calling Text Endpoint ")
        results = wh_client.search_index_endpoint(
            visionai_v1.SearchIndexEndpointRequest(
                index_endpoint=IMAGE_WAREHOUSE_ENDPOINT_NAME,
                text_query=QUERY,
            ),
        )
    else:
        # print(f"Calling Image Query {uploaded_file}")
        with open("search_image", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with open("search_image", "rb") as localfile:
            image_content = localfile.read()
        print(f"Calling Image Endpoint ")

        results = wh_client.search_index_endpoint(
            visionai_v1.SearchIndexEndpointRequest(
                index_endpoint=IMAGE_WAREHOUSE_ENDPOINT_NAME,
                image_query=visionai_v1.ImageQuery(
                    input_image=image_content,
                ),
            ),
        )
        # print(f"Called Image Query {results}")

    # print(f"received respnse {results}")

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
    # print(f"Images URIS size {len(uris)}")

    return uris


@st.cache_resource
def get_chat_client():
    # Initialize vertexai
    # vertexai.init(project=project_id, location=location)

    # Programmatically get an access token
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = transport.requests.Request()
    credentials.refresh(auth_request)

    # OpenAI client for Gemini-Flash-1.5
    client = openai.OpenAI(
        base_url=f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
        api_key=credentials.token,
    )
    return client


# chat_client = get_chat_client()


def get_response_chat(gemini_client, prompt="Why is the sky blue?"):
    response = gemini_client.chat.completions.create(
        model=gemini_15_flash._model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return response.choices[0].message.content
    except:
        return response


# This was generated by GenAI


def copy_file_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """Copies a file from a local drive to a GCS bucket.

    Args:
        local_file_path: The full path to the local file.
        bucket_name: The name of the GCS bucket to upload to.
        destination_blob_name: The desired name of the uploaded file in the bucket.

    Returns:
        None
    """

    import os
    from google.cloud import storage

    # Ensure the file exists locally
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"Local file '{local_file_path}' not found.")

    # Create a storage client
    storage_client = storage.Client()

    # Get a reference to the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a blob object with the desired destination path
    blob = bucket.blob(destination_blob_name)

    # Upload the file from the local filesystem
    content_type = ""
    if local_file_path.endswith(".html"):
        content_type = "text/html; charset=utf-8"

    if local_file_path.endswith(".json"):
        content_type = "application/json; charset=utf-8"

    if content_type == "":
        blob.upload_from_filename(local_file_path)
    else:
        blob.upload_from_filename(local_file_path, content_type=content_type)

    print(
        f"File '{local_file_path}' uploaded to GCS bucket '{bucket_name}' as '{destination_blob_name}.  Content-Type: {content_type}'."
    )


def download_from_gcs(destination_file_name, gcs_storage_bucket, object_name):
    # prompt: Write python code to download a blob from a gcs bucket.  do not use the requests method

    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_storage_bucket)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(object_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            object_name, gcs_storage_bucket, destination_file_name
        )
    )


# prompt: python to delete a file even if it does not exist


def delete_file(filename):
    try:
        os.remove(filename)
        print(f"File '{filename}' deleted successfully.")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")


def upload_to_gcs(image):
    """Uploads a file to GCS and returns the public URI."""
    # Create a client object
    image.save("input_image_for_video.png")

    storage_client = storage.Client(project=VEO_PROJECT_ID)
    # Get the bucket object
    bucket = storage_client.bucket(VEO_BUCKET)
    print(f"Calling upload to GCP file.name {bucket}")

    d = f"input_image_for_video.png"

    # d = bucket.blob(d)
    # Create a blob object with the filename
    blob = bucket.blob(d)
    # print(f"Calling upload to blob {blob}" )

    # Upload the file to GCS
    # blob.upload_from_file(image)
    blob.upload_from_filename(d)

    # Make the blob publicly accessible
    # gcs_uri =  blob.path_helper(VEO_BUCKET, d)

    # print(f"the Gcs uri for the image is {gcs_uri}" )
    # Return the public URI of the uploaded file
    # return gcs_uri


def restAPIHelper(url: str, http_verb: str, request_body: str) -> str:
    """Calls the Google Cloud REST API passing in the current users credentials"""

    import requests
    import google.auth
    import json

    # Get an access token based upon the current user
    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    access_token = creds.token

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + access_token,
    }

    if http_verb == "GET":
        response = requests.get(url, headers=headers)
    elif http_verb == "POST":
        response = requests.post(url, json=request_body, headers=headers)
    elif http_verb == "PUT":
        response = requests.put(url, json=request_body, headers=headers)
    elif http_verb == "PATCH":
        response = requests.patch(url, json=request_body, headers=headers)
    elif http_verb == "DELETE":
        response = requests.delete(url, headers=headers)
    else:
        raise RuntimeError(f"Unknown HTTP verb: {http_verb}")

    if response.status_code == 200:
        return json.loads(response.content)
        # image_data = json.loads(response.content)["predictions"][0]["bytesBase64Encoded"]
    else:
        error = f"Error restAPIHelper -> ' Status: '{response.status_code}' Text: '{response.text}'"
        raise RuntimeError(error)


def generate_video(image, prompt):
    """Calls text-to-video to create the video and waits for the output (which can be several minutes).  Saves the prompt/parameters with the vidoe.  Returns the outputted path."""

    full_output_gcs_path = f"gs://{VEO_BUCKET}/{VEO_OUTPUT_FOLDER}"
    model = "veo-001-preview-0815"
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{VEO_PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{model}:predictLongRunning"
    upload_to_gcs(image)
    gcs_uri = "gs://bagchi-genai-bb-veo-testing/input_image_for_video.png"

    request_body = {
        "instances": [
            {
                "prompt": prompt,
                "image": {
                    "gcsUri": gcs_uri,
                    "mimeType": "png/jpg",
                },
            }
        ],
        "parameters": {"storageUri": full_output_gcs_path, "aspectRatio": "16:9"},
    }

    rest_api_parameters = request_body.copy()

    #   print(f"url: {url}")
    print(f"request_body: {request_body}")
    json_result = restAPIHelper(url, "POST", request_body)
    # print(f"json_result: {json_result}")
    operation_name = json_result["name"]  # odd this is name

    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{VEO_PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{model}:fetchPredictOperation"

    request_body = {"operationName": operation_name}

    status = False

    while status == False:
        time.sleep(10)
        print(f"url: {url}")
        # print(f"request_body: {request_body}")
        json_result = restAPIHelper(url, "POST", request_body)
        # print(f"json_result: {json_result}")
        if "done" in json_result:
            status = bool(
                json_result["done"]
            )  # in the future might be a status of running
        else:
            print("Status not present.  Assuming not done...")

    # Get the filename of our video
    filename = json_result["response"]["generatedSamples"][0]["video"]["uri"]

    #   # Save our prompt (this was we know what we used to generate the video)
    #   json_filename = "text-to-video-prompt.json"
    #   with open(json_filename, "w") as f:
    #     f.write(json.dumps(rest_api_parameters))

    # get the random number directory from text-to-video
    #   text_to_video_output_directory = filename.replace(full_output_gcs_path,"")
    #   text_to_video_output_directory = text_to_video_output_directory.split("/")[1]
    #   text_to_video_output_directory

    # Write the prompt to the same path as our outputted video.  Saving the prompt allow us to know how to regenerate it (you should also save the seed and any other settings)
    #   copy_file_to_gcs(json_filename, storage_account, f"{output_gcs_path}/{text_to_video_output_directory}/{json_filename}")
    #   delete_file(json_filename)
    # filename = "Amorn"
    return filename


def initialize_backend():
    get_vertexai_session()
    # get_storage_url()

    story = generate_story(
        "Tell me a fancy story with a beauty and a beast within 200 words set in modern times",
        None,
    )
    # print(f"Loading a story: {story}")
    results = wh_client.search_index_endpoint(
        visionai_v1.SearchIndexEndpointRequest(
            index_endpoint=IMAGE_WAREHOUSE_ENDPOINT_NAME,
            text_query="an airplane",
        ),
    )
    # print(f"Loading a story: {results}")
