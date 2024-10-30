FROM python:3.9-slim
WORKDIR /app
COPY . ./

RUN pip install -r requirements.txt
RUN pip install visionai-0.0.6-py3-none-any.whl


CMD ["streamlit", "run", "home.py", "--server.enableCORS", "false", "--browser.serverAddress", "0.0.0.0", "--browser.gatherUsageStats", "false", "--server.port", "8080"]
