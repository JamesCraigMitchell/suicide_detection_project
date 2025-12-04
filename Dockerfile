FROM python:3.11-slim
WORKDIR /suicide_detection_project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "suicide_detection_classifier.py"]