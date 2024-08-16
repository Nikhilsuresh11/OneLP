from django.shortcuts import render
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from django.core.files.storage import FileSystemStorage
import os
import logging
from together import Together
import speech_recognition as sr
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import json
import PyPDF2
from sumy.parsers.plaintext import PlaintextParser
from dotenv import load_dotenv
from django.shortcuts import render
from pymongo import MongoClient
from db import db

load_dotenv()

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv('TOGETHER_API_KEY')
if not api_key:
    raise ValueError("TOGETHER_API_KEY is not set in the environment variables")


def home(request):
    return render(request, 'home.html')

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from pymongo import MongoClient
from django.contrib.auth.models import User

users_collection = db['users']

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password. Please try again.')

    return render(request, 'login.html')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Save user data in MongoDB
            user_data = {
                'username': user.username,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
                'password': user.password  # Store hashed password
            }
            users_collection.insert_one(user_data)

            return redirect('login')
        else:
            messages.error(request, "Invalid form submission. Please correct the errors.")
    else:
        form = UserCreationForm()

    return render(request, 'signup.html', {'form': form})


def contact(request):
    return render(request, 'contact.html')

def senti_home(request):
    return render(request, 'senti_home.html')

FRAME_RATE = 16000
CHANNELS = 1
MODEL_PATH = "path/to/vosk-model-en-us-0.22"

def voice(request):
    return render(request, 'voice.html')

def convert_audio(request):
    if request.method == 'POST' and request.FILES['audio']:
        audio_file = request.FILES['audio']

        # Initialize Vosk model and recognizer
        model = Model(model_name=MODEL_PATH)
        rec = KaldiRecognizer(model, FRAME_RATE)
        rec.SetWords(True)

        # Convert uploaded audio file to required format
        mp3 = AudioSegment.from_file(audio_file)
        mp3 = mp3.set_channels(CHANNELS)
        mp3 = mp3.set_frame_rate(FRAME_RATE)

        # Convert audio to text
        rec.AcceptWaveform(mp3.raw_data)
        result = rec.Result()

        # Convert JSON result to text
        text = json.loads(result)["text"]

        return JsonResponse({'text': text})
    else:
        return JsonResponse({'error': 'No audio file found'})


def summarize_url(request):
    if request.method == 'POST':
        url = request.POST.get('url')
        if url:
            try:
                parser = HtmlParser.from_url(url, Tokenizer("english"))
                doc = parser.document
                
                summarizer = TextRankSummarizer()
                summary_sentences = summarizer(doc, 50)  # Change the number of sentences as needed
                
                summary_text = ' '.join([str(sentence) for sentence in summary_sentences])
                
                return render(request, 'summary.html', {'summary': summary_text})
            except Exception as e:
                error_message = f"Error summarizing URL: {e}"
                return render(request, 'error.html', {'error_message': error_message})

    return render(request, 'summarize.html')



def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        return text
    


def summarize_document(request):
    if request.method == 'POST' and request.FILES.get('document'):
        uploaded_file = request.FILES['document']
        #logger.info(f"Uploaded file: {uploaded_file.name}")
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(filename)

        try:
            # Extract text from uploaded document
            document_text = extract_text_from_pdf(fs.path(filename))
            # logger.info(f"Extracted text: {document_text[:500]}...")  # Log first 500 chars

            # Format the user message for Together API
            user_message = {"role": "user", "content": document_text}

            # Generate completion using Together API
            response = client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",#meta-llama/Llama-3-70b-chat-hf
                messages=[user_message],
            )
            summary = response.choices[0].message.content
            # logger.info(f"Generated summary: {summary}")

            return render(request, 'summary.html', {'summary': summary, 'filename': uploaded_file.name})
        
        except Exception as e:
            error_message = f"Error summarizing document: {e}"
            #logger.error(error_message, exc_info=True)
            return render(request, 'error.html', {'error_message': error_message})

    #else:
        #logger.warning("No file uploaded or request method is not POST.")
    
    return render(request, 'summarize.html')


def summarize_text(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        if text:
            try:
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                summarizer = TextRankSummarizer()
                summary_sentences = summarizer(parser.document, 3)  # Change the number of sentences as needed
                summary_text = ' '.join([str(sentence) for sentence in summary_sentences])
                
                return render(request, 'summary.html', {'summary': summary_text})
            except Exception as e:
                error_message = f"Error summarizing text: {e}"
                return render(request, 'error.html', {'error_message': error_message})

    return render(request, 'summarize.html')





import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def sentiment(request):
    user_text = ""
    prediction = None
    category = ""
    sentiment_dict = {}
    summary = ""

    if request.method == 'POST':
        if 'user_text' in request.POST:
            user_text = request.POST.get('user_text')
            if user_text:
                # Processing user_text
                user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
                stop_words = set(stopwords.words('english'))
                tokens = word_tokenize(user_text)
                stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
                lemmatizer = WordNetLemmatizer() 
                lemmatized_output = [lemmatizer.lemmatize(word) for word in stopwords_removed]

                # Load TF-IDF vectorizer and model
                with open('models/tfidf_vectorizer.pkl', 'rb') as file:
                    tfidf_vectorizer = pickle.load(file)            
                with open('models/rf_model.pkl', 'rb') as file:
                    final_model = pickle.load(file)            

                # Transform user input using loaded vectorizer
                X_test_tfidf = tfidf_vectorizer.transform([' '.join(lemmatized_output)])

                # Apply model to make predictions
                prediction = final_model.predict(X_test_tfidf)

                # Perform sentiment analysis with VADER
                analyzer = SentimentIntensityAnalyzer()
                sentiment_dict = analyzer.polarity_scores(user_text)

                # Determine sentiment category
                if sentiment_dict['compound'] >= 0.05:
                    category = "Positive"
                elif sentiment_dict['compound'] <= -0.05:
                    category = "Negative"
                else:
                    category = "Neutral"

                # Prepare summary input and request summary from Together API
                summary_input = f"Input Text: {user_text} Sentiment Category: {category}"
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/Llama-3-70b-chat-hf",
                        messages=[{"role": "user", "content": summary_input}],
                    )
                    # Check response before accessing its content
                    if response and response.choices:
                        summary = response.choices[0].message.content
                    else:
                        summary = "Error: No valid response from the API"
                except Exception as e:
                    summary = f"Error: {str(e)}"

    return render(request, 'sentiment.html', {
        'user_text': user_text,
        'prediction': prediction,
        'category': category,
        'sentiment_dict': sentiment_dict,
        'summary': summary
    })

  # Assuming db.py is in the same directory

def create_blog(request):
    if request.method == 'POST':
        blog_title = request.POST.get('title')
        blog_tone = request.POST.get('tone')
        blog_word_count = request.POST.get('word_count')

        try:
            # Format the user message for the API
            user_message = {
                "role": "user",
                "content": f"Title: {blog_title}\n\nPlease write this blog in a {blog_tone} tone and keep it around {blog_word_count} words."
            }

            # Generate completion using Together API
            response = client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[user_message],
            )
            generated_blog = response.choices[0].message.content
            
            # Store the blog details in MongoDB
            blog_collection = db['blogs']  # Access the collection
            blog_data = {
                'title': blog_title,
                'tone': blog_tone,
                'word_count': blog_word_count,
                'generated_blog': generated_blog
            }
            blog_collection.insert_one(blog_data)  # Insert the data into the collection
            
            # Render the generated blog
            return render(request, 'blog.html', {'generated_blog': generated_blog, 'title': blog_title})

        except Exception as e:
            error_message = f"Error generating or storing blog: {e}"
            return render(request, 'error.html', {'error_message': error_message})

    return render(request, 'blog_form.html')


# Function to perform sentiment analysis using Together API



def perform_sentiment_analysis(text):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=[{"role": "user", "content": text}],
    )
    summary = response.choices[0].message.content
    return summary

# Function to determine sentiment category based on compound score
def get_sentiment_category(compound_score):
    if compound_score >= 0.05:
        return "Positive ✅"
    elif compound_score <= -0.05:
        return "Negative 🚫"
    else:
        return "Neutral ☑️"

# Main view for handling file upload and sentiment analysis
def sentiment_analysis(request):
    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            uploaded_file = request.FILES['csv_file']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_url = fs.url(filename)

            try:
                df = pd.read_csv(fs.path(filename))
            except Exception as e:
                return render(request, 'upload_csv.html', {'error_message': str(e)})

            context = {
                'columns': df.columns,
                'csv_uploaded': True,
                'csv_file_path': fs.path(filename)
            }
            return render(request, 'upload_csv.html', context)

        elif 'selected_column' in request.POST:
            selected_column = request.POST['selected_column']
            csv_file_path = request.POST['csv_file_path']

            try:
                df = pd.read_csv(csv_file_path)
            except Exception as e:
                return render(request, 'upload_csv.html', {'error_message': str(e)})

            texts = df[selected_column].astype(str).tolist()
            combined_text = ' '.join(texts)

            combined_text = re.sub('[%s]' % re.escape(string.punctuation), '', combined_text)
            stop_words = list(stopwords.words('english'))
            tokens = nltk.word_tokenize(combined_text)
            stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
            lemmatizer = WordNetLemmatizer()
            lemmatized_output = [lemmatizer.lemmatize(word) for word in stopwords_removed]

            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            X_test_tfidf = tfidf_vectorizer.transform(lemmatized_output)

            with open('models/rf_model.pkl', 'rb') as f:
                final_model = pickle.load(f)

            predictions = final_model.predict(X_test_tfidf)

            summary = perform_sentiment_analysis(combined_text)

            analyzer = SentimentIntensityAnalyzer()
            sentiment_dict = analyzer.polarity_scores(combined_text)

            sentiment_results = {
                'summary': summary,
                'sentiment_category': get_sentiment_category(sentiment_dict['compound']),
                'compound_score': sentiment_dict['compound'],
                'negative': sentiment_dict['neg'] * 100,
                'neutral': sentiment_dict['neu'] * 100,
                'positive': sentiment_dict['pos'] * 100,
            }

            context = {
                'columns': df.columns,
                'csv_uploaded': True,
                'sentiment_results': sentiment_results,
                'csv_file_path': csv_file_path
            }
            return render(request, 'upload_csv.html', context)

    return render(request, 'upload_csv.html')


client = Together(api_key=api_key)

def extract_text(file):
    text = ""
    if file.name.endswith('.pdf'):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ''
    elif file.name.endswith('.docx'):
        doc = Document(file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
    else:
        text = file.read().decode('utf-8')
    return text

def chatbot_response(conversation_history, document_text):
    formatted_messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversation_history]
    if document_text:
        formatted_messages.insert(0, {"role": "system", "content": "Document content: " + document_text})
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=formatted_messages,
    )
    return response.choices[0].message.content

@csrf_exempt
def chat_view(request):
    if "messages" not in request.session:
        request.session["messages"] = []
    if "document_text" not in request.session:
        request.session["document_text"] = ""

    if request.method == "POST":
        if 'csv_file' in request.FILES:
            uploaded_file = request.FILES['csv_file']
            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            with default_storage.open(file_path) as file:
                request.session["document_text"] = extract_text(file)
            return redirect('chat')
        elif 'user_input' in request.POST:
            user_input = request.POST.get('user_input')
            if user_input:
                request.session["messages"].append({"role": "user", "content": user_input})
                try:
                    response = chatbot_response(request.session["messages"], request.session["document_text"])
                    request.session["messages"].append({"role": "assistant", "content": response})
                except Exception as e:
                    return render(request, 'chat.html', {"error_message": f"Error: {e}"})
            return redirect('chat')

    return render(request, 'chat.html', {
        "messages": request.session.get("messages", []),
        "document_text": request.session.get("document_text", ""),
    })
