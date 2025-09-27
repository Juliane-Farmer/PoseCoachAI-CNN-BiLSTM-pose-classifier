FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY . .
RUN pip install -e .
ENV STREAMLIT_SERVER_HEADLESS=true STREAMLIT_SERVER_PORT=8501 POSECOACH_TTS=browser
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py"]
