from google.cloud import aiplatform
import base64
from google.cloud import storage
from google.protobuf import struct_pb2
import typing

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
    self.project = project


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
    response = self.client.predict(endpoint=endpoint, instances=instances)

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

  
def main():
  print("Enter Main")
  client = EmbeddingPredictionClient(project="bagchi-genai-bb")
  
  # Replace with your bucket name and folder name
  bucket_name = "bagchi-genai-bb"
  folder_name = "input_images"

  # Initialize a client
  storage_client = storage.Client()

  # Get the bucket
  bucket = storage_client.bucket(bucket_name)

  # List blobs (files) in the folder
  blobs = bucket.list_blobs(prefix=folder_name)

# Iterate through the blobs
  for blob in blobs:
    # Do something with each file
    print(f"Processing file: {blob.name}")
    # Example: Download the file
    if blob.name.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
      # blob.download_to_filename(f"/tmp/{blob.name}") 
      with blob.open('rb') as image_file:
          image_file_contents =image_file.read()
          response = client.get_embedding(image_bytes=image_file_contents)
          # print(f"Processing response: {response.image_embedding}")
          encoded_name = blob.name.encode(encoding = 'UTF-8', errors = 'strict')
          print(f"encoded_name: {encoded_name}")
          # write embedding to indexData.json file
          with open("indexData.json", "a") as f:
            f.write('{"id":"' + str(encoded_name) + '",')
            f.write('"embedding":[' + ",".join(str(x) for x in response[1]) + "]}")
            f.write("\n")
 
main()