import os
import gc
import pyodbc
import torch
import whisperx
from flask import Flask, render_template,send_from_directory, request, redirect, url_for
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from werkzeug.utils import secure_filename
import re
from datetime import datetime
import pypyodbc
import json
import time
import concurrent.futures
from pydub import AudioSegment
from flask_sqlalchemy import SQLAlchemy
from multiprocessing import Pool, cpu_count
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os



# Ensure VADER's lexicon is downloaded
nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'm4a', 'flac','opus'}

# Database connection parameters
DB_CONN_STR = 'Driver={ODBC Driver 17 for SQL Server};Server=CHETAN-SINGH\SQLEXPRESS;Database=ENGQMS;uid=sa;pwd=sa@123;'  # Uncomment this line for SQL Server Authentication
#DB_CONN_STR = 'Driver={ODBC Driver 17 for SQL Server};Server=Chetan-Singh;Database=ENGQMS;Trusted_Connection=yes;'  # Uncomment this line for SQL Server Authentication
print("Connection done")
# Function to test database connection
def test_db_connection():
    try:
        conn = pypyodbc.connect(DB_CONN_STR)
        conn.close()
        print("Database connection successful")
    except pypyodbc.Error as e:
        print(f"Database connection failed: {e}")

# Test the database connection
test_db_connection() 

""" DB_CONN_STR = os.getenv('DB_CONN_STR')
print(f"DB_CONN_STR: {DB_CONN_STR}")
app.config['SQLALCHEMY_DATABASE_URI'] = f'mssql+pyodbc:///?odbc_connect={DB_CONN_STR}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

print("Connection done")

# Function to test database connection
def test_db_connection():
    try:
        if DB_CONN_STR is None:
            raise ValueError("DB_CONN_STR environment variable is not set.")
        print(f"Attempting to connect with connection string: {DB_CONN_STR}")
        conn = pypyodbc.connect(DB_CONN_STR)
        conn.close()
        print("Database connection successful")
    except pypyodbc.Error as e:
        print(f"Database connection failed: {e}")

# Test the database connection
test_db_connection() """

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    # Ensure the filename has a dot and the extension is in the allowed set
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_audio_duration(filepath):
    audio = AudioSegment.from_file(filepath)
    return audio.duration_seconds

def get_file_size(filepath):
    # Get file size in bytes
    file_size_bytes = os.path.getsize(filepath)
    # Convert to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)
    return file_size_mb
@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        overall_start_time = time.time()
        file = request.files.get('file')
        language = request.form.get('language')
        keywords = request.form.get('keywords', '')
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_upload_start_time = time.time()
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_upload_end_time = time.time()
            file_upload_processing_time = (file_upload_end_time - file_upload_start_time) / 60  # in minutes
            
            keyword_list = keywords.split(',') if keywords else []
           
            transcription, sentiment_analysis, overall_sentiment_label, overall_sentiment_score, keyword_counts, diarization_result, text_with_times, pii_result,results = transcribe(filepath, language, keyword_list)
            print("Keyword counts in Flask route:", keyword_counts)  # Debugging line
            audio_duration = get_audio_duration(filepath)
            # Get file size
            file_size = get_file_size(filepath)
            print(f"Audio Duration: {audio_duration:.2f} seconds")  # Print the audio duration
            overall_end_time = time.time()
            overall_processing_time = (overall_end_time - overall_start_time) / 60  # in minutes
            print(f"File Upload Processing Time: {file_upload_processing_time:.2f} minutes")
            
            print(f"Overall Processing Time: {overall_processing_time:.2f} minutes")
             # Add cache-busting parameter
            cache_buster = datetime.now().strftime('%Y%m%d%H%M%S')
            return render_template('transcriptionpage.html', filepath=filepath.replace("\\", "/"), transcription=transcription, sentiment_analysis=sentiment_analysis,
                                   overall_sentiment_label=overall_sentiment_label, overall_sentiment_score=overall_sentiment_score,
                                   keyword_counts=keyword_counts, diarization_result=diarization_result, text_with_times=text_with_times, pii_result=pii_result,results=results)
    return render_template('transcriptionpage.html')

def count_keywords(text, keywords):
    keyword_counts = {}
    found_any_keyword = False
    
    for keyword in keywords:
        count = text.lower().count(keyword.lower())
        if count > 0:
            keyword_counts[keyword] = count
            found_any_keyword = True
    
    if not found_any_keyword:
       keyword_counts = None  # Set to None to indicate no keywords found
    
    return keyword_counts
def save_transcription_data(file_name, keyword_counts, positive_count, negative_count, results, call_id, file_status, error_msg, max_label, max_score, neutral_count, has_pii,audio_duration,total_speaking_time,file_size,detected_language):
    # Round max_score to 3 decimal places
    rounded_max_score = round(max_score, 2)
    file_size_mb=round(file_size, 3)
    #audio_durations= round(audio_duration, 2)
    complete_sentiment_str = json.dumps(results)
    print("Inserting the following values:")
    print(f"file_name: {file_name}")
    print(f"complete_sentiment_str: {complete_sentiment_str}")
    print(f"keyword_counts: {json.dumps(keyword_counts)}")
    print(f"positive_count: {positive_count}")
    print(f"negative_count: {negative_count}")
    print(f"max_score (before insert): {rounded_max_score}")
    print(f"call_id: {call_id}")
    print(f"file_status: {file_status}")
    print(f"error_msg: {error_msg}")
    print(f"Sentiments: {complete_sentiment_str}")
    print(f"max_label: {max_label}")
    print(f"neutral_count: {neutral_count}")
    print(f"has_pii: {has_pii}")
    print(f"audio_duration: {audio_duration}")
    print(f"last_end_time: {total_speaking_time}")
    print(f"fileszie: {file_size}")
    print(f"language: {detected_language}")
    try:
        conn = pypyodbc.connect(DB_CONN_STR)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Call_Transcribe (file_name, keyword_count, positive_word_count, negative_word_count, ixn_overall_sentiment_score, call_id, file_status, error_msg, sentiment_details, ixn_overall_sentiment_label, neutral_word_count, has_pii,total_recording_duration,actual_recording_duration,recording_file_size,lang_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?,?)
        """, (file_name,json.dumps(keyword_counts) if keyword_counts else None, positive_count, negative_count, rounded_max_score, call_id, file_status, error_msg, complete_sentiment_str, max_label, neutral_count, has_pii,audio_duration,total_speaking_time,file_size_mb,detected_language))
        conn.commit()
        print("Data inserted successfully.")
        #if keyword_counts is not None else None
    except pypyodbc.DatabaseError as db_err:
        print(f"Database error: {db_err}")
    except Exception as e:
        print(f"Error while inserting data: {e}")
    finally:
        cursor.close()
        conn.close()
def redact_pii(text):
    patterns = {
    'name': r'\b[A-Z][a-zA-Z]* [A-Z][a-zA-Z]*\b',  # Detects names with more flexibility in characters
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Detects email addresses
  
}
    #sentiment_pattern = r'\(Sentiment:.*?\)'
    redacted_text = text
    has_pii = 0
    for key, pattern in patterns.items():
        if re.search(pattern, redacted_text):
            has_pii = 1
        redacted_text = re.sub(pattern, '*****', redacted_text)

    return redacted_text, has_pii

def transcribe(filepath, language, keywords):
    is_cuda_available = torch.cuda.is_available()
    device = "cuda" if is_cuda_available else "cpu"
    compute_type = "float16" if is_cuda_available else "int8"

    print("CUDA available:", is_cuda_available)
    print("CUDA version:", torch.version.cuda if is_cuda_available else "N/A")
    whisperx_start_time = time.time()
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4  # Reduce if low on GPU memory
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    # Function to clear memory
    def clear_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Clear memory before loading the model
    clear_memory()

    audio = whisperx.load_audio(filepath)
    #model_path=r"C:\Users\Administrator\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3\snapshots\edaa852ec7e145841d8ffdb056a99866b5f0a478"
    #model = whisperx.load_model(model_path, device=device, compute_type=compute_type)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # Transcribe in a separate thread
    #with concurrent.futures.ThreadPoolExecutor() as executor:
        #transcription_future = executor.submit(model.transcribe, audio, batch_size=batch_size)
        
        #result = transcription_future.result()
    # Optimize thread pool size based on your CPU cores
   # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #transcription_future = executor.submit(model.transcribe, audio, batch_size=batch_size)
       #result = transcription_future.result() 
    result = model.transcribe(audio, batch_size=batch_size,language='hi')
    print("Transcription result:", result)
    
    print("Transcription result:", result)
    print("Detected language:", result["language"])
    # Save the detected language in a variable
    detected_language_code = result["language"]

    # Mapping of ISO 639-1 language codes to full names
    language_mapping = {
        "en": "English",
        "hi": "Hindi",
        
        # Add more languages as needed
    }

    # Get the full language name
    detected_language = language_mapping.get(detected_language_code, "Unknown")
    print("language name",detected_language)
    print("Model segments:", result["segments"])
    segments = result["segments"]
    if segments:
        last_segment = segments[-1]
        last_end_time = last_segment.get("end", None)
        print(f"Last segment ends at: {last_end_time}")
    else:
        print("No segments found.")
    
    file_size = get_file_size(filepath)
    print("filesize",file_size)
    whisperx_end_time = time.time()
    whisperx_processing_time = (whisperx_end_time - whisperx_start_time) / 60
    print(f"WhisperX Processing Time: {whisperx_processing_time:.2f} minutes")
    text_with_times = [
        {"text": segment["text"], "start": segment["start"], "end": segment["end"]}
        for segment in result["segments"]
    ]

    print("Text with times:", text_with_times)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if result["language"] == "ur":
        result["language"] = "hi"
        for segment in result["segments"]:
            segment["language"] = "hi"
    start_time = time.time()

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
    #with concurrent.futures.ThreadPoolExecutor() as executor:
        #diarization_future = executor.submit(diarize_model, audio, min_speakers=2, max_speakers=5)
        
        #diarization_result = diarization_future.result()
    #with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #diarization_future = executor.submit(diarize_model, audio, min_speakers=2, max_speakers=5)
        #diarization_result = diarization_future.result()
    # Track the end time for diarization
    diarization_result = diarize_model(audio, min_speakers=2, max_speakers=3)
    end_time = time.time()

    # Calculate total diarization time
    diarization_time = end_time - start_time

    print("Diarization result:", diarization_result)
    print(f"Total time taken for diarization: {diarization_time:.2f} seconds")
    def custom_assign_word_speakers(diarization_result, transcription_result):
        import pandas as pd
        
        if isinstance(diarization_result, pd.DataFrame):
            diarization_segments = diarization_result
        else:
            raise ValueError("Unexpected diarization_result format")
        
        assigned_result = transcription_result.copy()
        
        for segment in assigned_result["segments"]:
            for word in segment["words"]:
                word_start = word["start"]
                word_end = word["end"]
                word_speaker = None
                
                for idx, diarization_segment in diarization_segments.iterrows():
                    segment_start = diarization_segment["start"]
                    segment_end = diarization_segment["end"]
                    segment_speaker = diarization_segment["speaker"]
                    
                    # Check if word overlaps with diarization segment
                    if segment_start <= word_start < segment_end or segment_start < word_end <= segment_end or (word_start < segment_start and word_end > segment_end):
                        word_speaker = segment_speaker
                        break
                
                if word_speaker is None:
                    word_speaker = "Unknown"
                    print(f"Word '{word['word']}' (start: {word_start}, end: {word_end}) assigned to 'Unknown'")
                
                word["speaker"] = word_speaker
        
        return assigned_result

    # Call the custom function
    #result = custom_assign_word_speakers(diarization_result, result)
    result = whisperx.assign_word_speakers(diarization_result, result)

    #sentiment_analyzer_en = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    #sentiment_analyzer_hi = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    label_mapping = ['negative', 'neutral', 'positive']
    # Ensure VADER's lexicon is downloaded
    # Ensure VADER's lexicon is downloaded
    nltk.download('vader_lexicon')

    # Initialize VADER SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Load a custom lexicon if needed
    def load_custom_lexicon(file_path):
        custom_lexicon = {}
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return custom_lexicon
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        word, score = line.split(':')
                        custom_lexicon[word] = float(score)
        except Exception as e:
            print(f"Error reading file: {e}")
        return custom_lexicon

    # Add custom lexicon to VADER
    def add_custom_words(sia, lexicon):
        for word, score in lexicon.items():
            sia.lexicon[word] = score

    file_path = 'custom_lexicon.txt'
    custom_lexicon = load_custom_lexicon(file_path)
    add_custom_words(sia, custom_lexicon)

    def format_output(result, include_sentiment=True, include_vader_sentiment=True):
        formatted_output = ""
        plain_text = ""
        current_speaker = None
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_positive_score = 0
        total_negative_score = 0
        total_neutral_score = 0

        total_speaking_time = 0
        silent_time = 0
        previous_end_time = None
        audio_start_time = 0

        segments = result["segments"]

        for segment in segments:
            words = segment['words'][0]
            speaker = words.get('speaker', None)
            text = segment['text'].strip()
            plain_text += text + " "

            start_time, end_time = segment["start"], segment["end"]
            speaking_duration = end_time - start_time
            total_speaking_time += speaking_duration

            if previous_end_time is None:
                silent_time += start_time - audio_start_time
            elif start_time > previous_end_time:
                silent_time += start_time - previous_end_time
            previous_end_time = end_time

            # Sentiment Analysis (Current Model)
            inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                logits = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            sentiment_score, sentiment_label_id = torch.max(probabilities, dim=1)
            sentiment_score = sentiment_score.item()
            sentiment_label = label_mapping[sentiment_label_id.item()]

            # Count sentiments
            if sentiment_label == "positive":
                positive_count += 1
                total_positive_score += sentiment_score
            elif sentiment_label == "negative":
                negative_count += 1
                total_negative_score += sentiment_score
            else:
                neutral_count += 1
                total_neutral_score += sentiment_score

            # VADER Sentiment Analysis
            if include_vader_sentiment:
                vader_scores = sia.polarity_scores(text)
                vader_sentiment = (
                    "Positive" if vader_scores["compound"] > 0 else
                    "Negative" if vader_scores["compound"] < 0 else
                    "Neutral"
                )
                vader_details = f" (VADER: {vader_sentiment}, Score: {vader_scores['compound']:.2f})"
            else:
                vader_details = ""

            # Speaker Segmentation
            if speaker != current_speaker:
                if current_speaker is not None:
                    formatted_output += "\n\n"
                current_speaker = speaker
                formatted_output += f"{start_time:.2f} {speaker}: "

            formatted_output += f"[{sentiment_label}-{start_time}]{vader_details} {text} " if include_sentiment else f"{text} "

        return (
            formatted_output.strip(), plain_text.strip(), positive_count, negative_count,
            neutral_count, total_positive_score, total_negative_score, total_neutral_score, total_speaking_time
        )

    # Call the function
    include_sentiment = True
    include_vader_sentiment = True

    formatted_result, plain_text, positive_count, negative_count, neutral_count, total_positive_score, total_negative_score, total_neutral_score, total_speaking_time = format_output(
        result, include_sentiment, include_vader_sentiment
    )
    overall_sentiment_scores = {
        "positive": total_positive_score,
        "negative": total_negative_score,
        "neutral": total_neutral_score
    }
    """ overall_sentiment_label = max(overall_sentiment_scores, key=overall_sentiment_scores.get)
    overall_sentiment_score = overall_sentiment_scores[overall_sentiment_label] """
    print("overallScore",overall_sentiment_scores)
    
    average_positive_sentiment = total_positive_score / positive_count if positive_count > 0 else 0
    average_negative_sentiment = total_negative_score / negative_count if negative_count > 0 else 0
    average_neutral_sentiment = total_neutral_score / neutral_count if neutral_count > 0 else 0

    audio_duration = get_audio_duration(filepath)
    print(f"Audio Duration: {audio_duration:.2f} seconds")

    keyword_counts = count_keywords(plain_text, keywords)
    print(keyword_counts)
    print("Formatted Result:")
    print(formatted_result)
    print("Positive Sentiment Score:", total_positive_score)
    print("Negative Sentiment Score:", total_negative_score)
    print("Neutral Sentiment Score:", total_neutral_score)
 
    print("Positive Count:", positive_count)
    print("Negative Count:", negative_count)
    print("Neutral Count:", neutral_count)
    print("Average Positive Sentiment:", average_positive_sentiment)
    print("Average Negative Sentiment:", average_negative_sentiment)
    print("Average Neutral Sentiment:", average_neutral_sentiment)
  

    

    def load_custom_lexicon(file_path):
        """
        Load a custom lexicon from a text file with debugging.
        """
        custom_lexicon = {}
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return custom_lexicon

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if not lines:
                    print("File is empty.")
                for line in lines:
                    line = line.strip()  # Remove leading/trailing spaces
                    if line:  # Skip empty lines
                        print(f"Reading line: {line}")  # Debugging: show the line being read
                        try:
                            word, score = line.split(':')
                            custom_lexicon[word] = float(score)  # Convert score to float
                        except ValueError as e:
                            print(f"Error parsing line: {line}. Error: {e}")
        except Exception as e:
            print(f"Error reading file: {e}")
        return custom_lexicon

    def add_custom_words(sia, lexicon):
        """
        Add custom words to VADER's SentimentIntensityAnalyzer.

        Args:
            sia (SentimentIntensityAnalyzer): VADER SentimentIntensityAnalyzer instance.
            lexicon (dict): Custom lexicon dictionary.
        """
        for word, score in lexicon.items():
            sia.lexicon[word] = score

    # Path to the custom lexicon text file
    file_path = 'custom_lexicon.txt'

    # Step 1: Load the custom lexicon
    custom_lexicon = load_custom_lexicon(file_path)
    print("Loaded Custom Lexicon:", custom_lexicon)

    # Step 2: Initialize VADER SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Step 3: Add custom words to VADER's lexicon
    add_custom_words(sia, custom_lexicon)

    # Step 4: Analyze a text using the updated lexicon
    #text = "this is not a bad things"
    scores = sia.polarity_scores(plain_text)

    # Extract positive, negative, neutral words with their scores
    positive_words = [(word, sia.lexicon.get(word.lower(), 0)) for word in plain_text.split() if sia.lexicon.get(word.lower(), 0) > 0]
    negative_words = [(word, sia.lexicon.get(word.lower(), 0)) for word in plain_text.split() if sia.lexicon.get(word.lower(), 0) < 0]
    neutral_words = [(word, sia.lexicon.get(word.lower(), 0)) for word in plain_text.split() if sia.lexicon.get(word.lower(), 0) == 0]

    # Display Results
    print("Positive Words and Scores:", positive_words)
    print("Negative Words and Scores:", negative_words)
    print("Neutral Words and Scores:", neutral_words)
    print("Overall Sentiment Score:", scores['compound'])
    print("Overall Sentiment:", "Positive" if scores['compound'] > 0 else "Negative" if scores['compound'] < 0 else "Neutral")

    diarization_results = format_output(result)[0]
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("plain_text",plain_text)
    
    #if result.get("language") == "hi":
        
       # overall_sentiment = sentiment_analyzer_hi(plain_text)[0]
    #else:
        #overall_sentiment = sentiment_analyzer_en(plain_text)[0]
     # Tokenize with error handling
  # Tokenize with truncation
    try:
        inputs = tokenizer(plain_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        #print("Tokenized inputs:", inputs)  # Debug print

        with torch.no_grad():
            logits = model(**inputs).logits

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Define label mapping
        label_mappings = ['negative', 'neutral', 'positive']
        results = {label: prob.item() for label, prob in zip(label_mappings, probabilities[0])}
        max_label = max(results, key=results.get)
        max_score = results[max_label]
        max_result = (max_label, max_score)
        print("Max Result:", max_result)
        print("max_label",max_label)
        # Store both label and score in a single variable
        max_result = (max_label, max_score)
        print("max_result",max_result)
        # Print the result
        print(f"Label with highest score: {max_result[0]}, Score: {max_result[1]}")
        print("text data",results)
    except Exception as e:
        print("Error during model processing:", e)
        
    #print("Sentiment Scores Structure:", overall_sentiment)
    overall_sentiment_label = max_label
    overall_sentiment_score = max_score
    
    
    
    sentiment_summary = (
        f"Sentiment: {overall_sentiment_label} \n"
        #f"Positive Count: {positive_count}\n"
        #f"overall result: {results}\n"
        #f"Negative Count: {negative_count}\n"
        #f"Neutral Count: {neutral_count}\n"
        #f"Average Positive Sentiment: {average_positive_sentiment:.2f}\n"
        #f"Average Negative Sentiment: {average_negative_sentiment:.2f}\n"
        #f"Average Neutral Sentiment: {average_neutral_sentiment:.2f}\n" 
    )
    complete_sentiment = f"{overall_sentiment_label} (Score: {overall_sentiment_score:.2f})"
    print(complete_sentiment)
    print("sentiment summary",sentiment_summary)
    # Perform sentiment analysis on the plain text
    
    #pii_result = [redact_pii(result) for result in diarization_results]

    # Print the redacted results for the UI
    #for result in pii_result:
        #print(result)
    pii_result, has_pii = redact_pii(diarization_results)
    output_file = os.path.splitext(filepath)[0] + ".txt"

    # Combine diarization results and plain text, adding keyword 8893
    combined_text = f"{diarization_results}\n~\n{plain_text}\n"
    # Get the base name of the file
    base_name = os.path.basename(filepath)

    # Split the base name to remove the extension
    file_name = os.path.splitext(base_name)[0]

    print(file_name)
    print("filepath",file_name)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(combined_text)
    call_id =file_name.split('-')[10]  # You can define how to get the call_id
    file_status = 'success'
    error_msg = None
    save_transcription_data(
    file_name, 
    keyword_counts, 
    positive_count, 
    negative_count, 
    results, 
    call_id, 
    file_status, 
    error_msg, 
    max_label, 
    max_score, 
    neutral_count, 
    has_pii,audio_duration,total_speaking_time,file_size,detected_language
)
    return plain_text, sentiment_summary, overall_sentiment_label, overall_sentiment_score, keyword_counts, diarization_results, text_with_times, pii_result,results

if __name__ == '__main__':
    app.run(debug=False)
