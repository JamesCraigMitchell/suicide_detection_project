FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download(['stopwords', 'wordnet', 'punkt_tab'])"
COPY . .
CMD ["python", "suicide_detection_classifier.py"]