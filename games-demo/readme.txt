#local runtime
pip install virtualenv
 brew install python@3.9
python3.9 -m venv .
gcloud config set project bagchi-genai-bb
gcloud auth application-default login
gcloud auth application-default set-quota-project bagchi-genai-bb
pip install -r requirements.txt
wget https://github.com/google/visionai/releases/download/v0.0.6/visionai-0.0.6-py3-none-any.whl
pip3 install visionai-0.0.6-py3-none-any.whl

streamlit run home.py \
  --browser.serverAddress=localhost \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.port 8080

COPY visionai-0.0.6-py3-none-any.whl . 
RUN pip install visionai-0.0.6-py3-none-any.whl



gcloud builds submit --tag gcr.io/bagchi-genai-bb/gaming-3d-assset
gcloud run deploy gaming-assistant-app --image gcr.io/bagchi-genai-bb/gaming-3d-assset --platform managed  --allow-unauthenticated  --region us-central1

sudo ls -l /usr/local/bin | grep '../Library/Frameworks/Python*' | awk '{print $9}' | tr -d @ | xargs rm -f

sudo -H python3 -m ensurepip


# Embedding Generation
python3 images-to-embedding.py 
gsutil cp indexData.json gs://bagchi-genai-bb/images/indexData.json


gcloud ai indexes create \
  --metadata-file=index-metadata.json \
  --display-name=GamesDemo-MultiModal-Embeddings \
  --project=bagchi-genai-bb \
  --region=us-central1


curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d @request.json \
    "https://us-central1-aiplatform.googleapis.com/v1/projects/bagchi-genai-bb/locations/us-central1/indexEndpoints"

gcloud ai index-endpoints deploy-index 340703469075693568 \
  --deployed-index-id= 9118078405332434944\
  --display-name=games-demo-images-embedding \
  --index= \
  --project=bagchi-genai-bb \
  --region=us-central1

#Image warehouse

  curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @request_image_wh.json \
     "https://warehouse-visionai.googleapis.com/v1/projects/104454103637/locations/us-central1/indexEndpoints/games-search-endpoint-demo:searchIndexEndpoint"