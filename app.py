import pdfminer
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import requests
import io
import os
import nltk

# Set up NLTK data path
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

app = Flask(__name__)

# URL of the PDF on GitHub
pdf_url = "https://raw.githubusercontent.com/Soubarnikaentrans/Glitch/main/report.pdf"

def download_pdf_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    return io.BytesIO(response.content)

def extract_text_from_pdf(pdf_stream):
    text = extract_text(pdf_stream)
    return text

def preprocess_text(text):
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    tokens = [[token.lower() for token in sentence] for sentence in tokens]
    tokens = [[token for token in sentence if token not in string.punctuation] for sentence in tokens]
    tokens = [[token for token in sentence if token not in stopwords.words('english')] for sentence in tokens]
    return [" ".join(sentence) for sentence in tokens], sentences

# Fetch the PDF from GitHub and process it
pdf_stream = download_pdf_from_url(pdf_url)
pdf_text = extract_text_from_pdf(pdf_stream)
processed_text, original_sentences = preprocess_text(pdf_text)

if not processed_text:
    raise Exception("Text extraction or preprocessing failed. Check the PDF file and path.")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_text)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = retrieve_information(user_input)
    return jsonify({"response": response})

def retrieve_information(query):
    query_vec = vectorizer.transform([query.lower()])
    results = (X * query_vec.T).toarray()
    relevant_indices = np.argsort(results.flatten())[::-1]
    top_n = 3  # Number of top results to return
    relevant_sentences = [original_sentences[i] for i in relevant_indices[:top_n]]
    return " ".join(relevant_sentences)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
