from flask import Flask, request
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")


application = Flask(__name__)

# Load advanced summarization models
summarizers = {
    "bart": pipeline("summarization", model="facebook/bart-large-cnn"),
    "t5": pipeline("summarization", model="t5-small"),
    "pegasus": pipeline("summarization", model="google/pegasus-xsum")
}

@application.get('/summary')
def summary_api():
    """
    Summarizes the transcript of a YouTube video using advanced models.
    
    Parameters:
    - url (str): The YouTube video URL.
    - max_length (int, optional): The maximum length of the summary (default: 150).
    - model (str, optional): The summarization model to use (bart, t5, pegasus).

    Returns:
    - str: The summarized transcript.
    """
    url = request.args.get('url', '')
    max_length = int(request.args.get('max_length', 150))
    model_name = request.args.get('model', 'bart')  # Default to BART
    video_id = url.split('=')[1]

    try:
        transcript = get_transcript(video_id)
    except:
        return "No subtitles available for this video", 404

    # Extractive summarization for large transcripts
    if len(transcript.split()) > 3000:
        summary = extractive_summarization(transcript)
    else:
        summary = abstractive_summarization(transcript, max_length, model_name)

    return summary, 200


def get_transcript(video_id):
    """Fetches and concatenates the transcript of a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        raise e

    transcript = ' '.join([d['text'] for d in transcript_list])
    return transcript


def abstractive_summarization(transcript, max_length, model_name):
    """
    Uses an advanced transformer model for summarization.
    
    - Supports BART, T5, and Pegasus.
    """
    if model_name not in summarizers:
        return "Invalid model name. Choose from: bart, t5, pegasus.", 400

    summarizer = summarizers[model_name]
    summary = ''
    
    # Process in chunks to handle long text
    for i in range(0, (len(transcript) // 1000) + 1):
        chunk = transcript[i * 1000:(i+1) * 1000]
        summary_text = summarizer(chunk, max_length=max_length, do_sample=False)[0]['summary_text']
        summary += summary_text + ' '

    return summary


def extractive_summarization(transcript):
    """
    Uses Latent Semantic Analysis (LSA) for extractive summarization.
    """
    sentences = sent_tokenize(transcript)

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)

    svd = TruncatedSVD(n_components=1, random_state=42)
    svd.fit(X)
    components = svd.transform(X)

    ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
    
    num_sentences = int(0.4 * len(sentences))
    selected_sentences = sorted(ranked_sentences[:num_sentences])

    summary = " ".join([sentences[idx] for idx in selected_sentences])
    return summary


if __name__ == '__main__':
    application.run(debug=True)
