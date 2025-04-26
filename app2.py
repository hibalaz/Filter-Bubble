import streamlit as st
import whisper
import os
import time
import json
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import subprocess
import traceback
import hashlib
import logging
import tempfile
from models import load_models, classify_political, classify_bias, identify_topics
from tiktok import TikTokAPIClient

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('filter_bubble')

# Page configuration
st.set_page_config(
    page_title="Filter Bubble Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replace the CSS section in your app.py with this improved dark theme code:

# Dark Theme CSS with better contrast and visibility
# Enhanced Dark Theme with Modern UI Elements
st.markdown("""
<style>
    /* Base theme improvements */
    .stApp {
        background-color: #111827;
        color: #E5E7EB;
    }
    
    /* Header styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #FFFFFF;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        letter-spacing: -0.5px;
    }
    
    /* Welcome message */
    p:has(+ div[data-testid="stHorizontalBlock"]) {
        font-size: 1.2rem;
        font-weight: 500;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }
    
    /* Cards and containers */
    .card, div[data-testid="stForm"], div[data-testid="stExpander"] {
        background-color: #1F2937;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #374151;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover, div[data-testid="stExpander"]:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    /* Tabs styling */
    button[data-baseweb="tab"] {
        background-color: #1F2937;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        margin-right: 4px;
        font-weight: 600;
        font-size: 16px;
        border: 1px solid #374151;
        border-bottom: none;
        color: #9CA3AF;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2563EB;
        border-color: #2563EB;
        color: white;
    }
    
    /* Section headers */
    h2, h3, .subheader, div[data-testid="stExpander"] > div[role="button"] > div > div:first-child {
        color: #FFFFFF;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #374151;
    }
    
    /* Buttons styling */
    button, button:focus, button:active, div[data-testid="stFormSubmitButton"] > button {
        background-color: #2563EB !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.3) !important;
        text-transform: none !important;
        letter-spacing: 0.2px !important;
    }
    
    button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Specific success button styling */
    button[kind="primary"], .success-btn {
        background-color: #10B981 !important;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3) !important;
    }
    
    button[kind="primary"]:hover, .success-btn:hover {
        background-color: #059669 !important;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Delete button styling */
    button[id^="delete_"] {
        background-color: #EF4444 !important;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3) !important;
    }
    
    button[id^="delete_"]:hover {
        background-color: #DC2626 !important;
        box-shadow: 0 4px 8px rgba(239, 68, 68, 0.4) !important;
    }
    
    /* Input field styling */
    input, textarea, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div {
        background-color: #374151 !important;
        border-color: #4B5563 !important;
        border-radius: 8px !important;
        color: white !important;
        caret-color: white !important;
    }
    
    input:focus, textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }
    
    /* Placeholder text */
    input::placeholder, textarea::placeholder {
        color: #9CA3AF !important;
    }
    
    /* File uploader styling */
    div[data-testid="stFileUploader"] {
        background-color: #1F2937;
        border: 2px dashed #4B5563;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #2563EB;
        background-color: rgba(37, 99, 235, 0.05);
    }
    
    div[data-testid="stFileUploader"] > div > div > p {
        color: #9CA3AF !important;
    }
    
    /* Browse files button */
    div[data-testid="stFileUploader"] button {
        background-color: #374151 !important;
        border: 1px solid #4B5563 !important;
        color: white !important;
    }
    
    div[data-testid="stFileUploader"] button:hover {
        background-color: #4B5563 !important;
    }
    
    /* Progress bar styling */
    div[data-testid="stProgress"] > div > div {
        background-color: #2563EB !important;
    }
    
    /* Video list styling */
    div[data-testid="stExpander"] {
        margin-bottom: 0.75rem;
    }
    
    /* Alerts and info boxes */
    div.element-container div[data-testid="stAlert"] {
        background-color: #1F2937;
        border-left: 4px solid;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    
    div[data-testid="stAlert"][kind="info"] {
        border-left-color: #2563EB;
    }
    
    div[data-testid="stAlert"][kind="warning"] {
        border-left-color: #F59E0B;
    }
    
    div[data-testid="stAlert"][kind="error"] {
        border-left-color: #EF4444;
    }
    
    div[data-testid="stAlert"][kind="success"] {
        border-left-color: #10B981;
    }
    
    div[data-testid="stAlert"] > div {
        color: #E5E7EB !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #374151;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: white;
        font-size: 1.5rem;
        padding: 1rem 0;
    }
    
    section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"] {
        background-color: #334155 !important;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1E293B;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4B5563;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6B7280;
    }
    
    /* Animation for hover effects */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(37, 99, 235, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(37, 99, 235, 0);
        }
    }
    
    /* Logout button styling */
    button:has(+ img[alt="Logout"]), div.element-container:has(> div.row-widget.stButton > button:contains("Logout")) button {
        background-color: #4B5563 !important;
        position: absolute;
        top: 1rem;
        right: 1rem;
    }
    
    /* Fix for light/dark theme toggle */
    button[data-testid="baseButton-headerNoPadding"] {
        color: white !important;
    }
    
    /* TikTok specific styling */
    div.element-container:has(h2:contains("Connect to TikTok")) {
        margin-top: 2rem;
    }
    
    /* Logo styling for branding */
    img {
        filter: drop-shadow(0 4px 3px rgba(0, 0, 0, 0.07)) drop-shadow(0 2px 2px rgba(0, 0, 0, 0.06));
    }
    
    /* Analysis results styling */
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #2563EB;
        line-height: 1;
        margin: 1rem 0;
    }
    
    .risk-high {
        color: #EF4444;
    }
    
    .risk-moderate {
        color: #F59E0B;
    }
    
    .risk-low {
        color: #10B981;
    }
    
    /* Make sure video items have clear separation */
    .video-item, div[data-testid="stExpander"] {
        border-bottom: 1px solid #374151;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Improved layout for "Your Videos" section */
    h3:contains("Your Videos") {
        font-size: 1.5rem;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2563EB;
        color: white;
    }
    
    /* Analyze button styling */
    div:has(button:contains("Analyze")) {
        margin-top: 2rem;
        text-align: center;
    }
    
    button:contains("Analyze") {
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    }
    
    button:contains("Analyze"):hover {
        background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Fix coloring of descriptions */
    .stMarkdown p {
        color: #9CA3AF;
    }
    
    /* Ensure expandable items are clearly clickable */
    div[data-testid="stExpander"] > div[role="button"] {
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# Configure directories
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
USER_DATA_FILE = os.path.join(DATA_DIR, 'user_data.json')
ERROR_LOG_FILE = os.path.join(DATA_DIR, 'error_log.txt')

# Check if ffmpeg is available on the system
def is_ffmpeg_available():
    """Check if ffmpeg is available on the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Helper function to extract audio from video
def extract_audio_from_video(video_path):
    """
    Extract audio from video file using ffmpeg
    Returns the path to the extracted audio file
    """
    try:
        # Create a temporary file for the audio
        temp_dir = tempfile.gettempdir()
        video_filename = os.path.basename(video_path)
        audio_filename = f"{os.path.splitext(video_filename)[0]}.mp3"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        # Use ffmpeg to extract audio
        command = [
            'ffmpeg',
            '-i', video_path,  # Input video file
            '-q:a', '0',       # Highest audio quality
            '-map', 'a',       # Extract only audio
            '-y',              # Overwrite output file if it exists
            audio_path         # Output audio file
        ]
        
        # Execute the command
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise Exception(f"Failed to extract audio from video: {str(e)}")
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise Exception(f"Error extracting audio: {str(e)}")

def transcribe_video(video_path):
    """Transcribe video using local Whisper model with ffmpeg for audio extraction"""
    try:
        # Check if ffmpeg is available
        if not is_ffmpeg_available():
            return "FFmpeg is not installed or not in PATH. Please install FFmpeg to enable transcription."
            
        # Extract audio from video
        audio_path = extract_audio_from_video(video_path)
        
        # Get model size from session state
        model_size = st.session_state.get('WHISPER_MODEL', 'base')
        
        # Add a progress message since model loading can take time
        logging.info(f"Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        
        # Transcribe the audio with the local model
        logging.info("Transcribing audio...")
        result = model.transcribe(audio_path)
        transcript = result["text"]
        
        # Clean up the temporary audio file
        try:
            os.remove(audio_path)
        except:
            pass  # Ignore cleanup errors
            
        return transcript
        
    except Exception as e:
        log_error(f"Error transcribing video: {str(e)}", e)
        
        # Fallback message with more detail for troubleshooting
        error_message = str(e)
        if "ffmpeg" in error_message.lower():
            return "Transcription failed: ffmpeg is not installed or not in PATH. Please install ffmpeg to enable transcription."
        else:
            return f"Transcription failed. Error: {error_message}"

# Helper functions for data persistence
def save_user_data():
    """Save user data to a JSON file for persistence"""
    try:
        with open(USER_DATA_FILE, 'w') as f:
            # Convert session state users dict to serializable format
            serializable_users = {}
            for username, user_data in st.session_state.users.items():
                # Deep copy to avoid modifying original
                user_copy = user_data.copy()
                # Handle non-serializable objects if any
                if 'videos' in user_copy:
                    for video in user_copy['videos']:
                        # Ensure no non-serializable objects
                        if 'file_obj' in video:
                            del video['file_obj']
                
                serializable_users[username] = user_copy
            
            json.dump(serializable_users, f)
        return True
    except Exception as e:
        log_error(f"Failed to save user data: {str(e)}", e)
        return False
        
def load_user_data():
    """Load user data from JSON file if it exists"""
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        log_error(f"Failed to load user data: {str(e)}", e)
        return {}

def log_error(error_message, error=None):
    """Log errors to a file for debugging"""
    try:
        logger.error(error_message)
        if error:
            logger.error(traceback.format_exc())
    except:
        pass  # Silent fail for logging errors


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1251/1251689.png", width=100)
    st.title("Filter Bubble Analyzer")
    
    # Whisper model configuration
    with st.expander("🔧 Whisper Configuration"):
        whisper_model = st.selectbox(
            "Whisper Model",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # "base" as default
            help="Larger models are more accurate but slower and require more memory"
        )
        
        if 'WHISPER_MODEL' not in st.session_state:
            st.session_state.WHISPER_MODEL = whisper_model
        elif whisper_model != st.session_state.WHISPER_MODEL:
            st.session_state.WHISPER_MODEL = whisper_model
            st.success(f"Whisper model set to: {whisper_model}")
    
    # FFmpeg installation guide (keep this as is)
    with st.expander("⚙️ Dependencies"):
        st.markdown("""
        This app requires **ffmpeg** for video processing:
        
        **Ubuntu/Debian:**
        ```
        sudo apt-get install ffmpeg
        ```
        
        **macOS:**
        ```
        brew install ffmpeg
        ```
        
        **Windows:**
        Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use:
        ```
        choco install ffmpeg
        ```
        """)
        
        # Add whisper installation guide
        st.markdown("""
        The app now uses the **whisper** Python package:
        
        ```
        pip install -U openai-whisper
        ```
        
        Note: This will download PyTorch and other dependencies if not already installed.
        """)
    
    # Keep the About section as is
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Filter Bubble Analyzer** helps you understand your social media content consumption patterns.
        
        Features:
        - Analyze TikTok feeds
        - Upload and analyze videos
        - Identify political bias and echo chambers
        - Get personalized recommendations
        
        This app was created to promote media literacy and diverse content consumption.
        """)
    
    # App version
    st.caption("Version 1.0.0 | © 2023")

if 'WHISPER_MODEL' not in st.session_state:
    st.session_state.WHISPER_MODEL = "base"  # Default model size

# Initialize TikTok API client
@st.cache_resource
def load_tiktok_api():
    return TikTokAPIClient()

tiktok_api = load_tiktok_api()

# Load ML models
@st.cache_resource
def load_ml_models():
    with st.spinner("Loading machine learning models..."):
        return load_models()

ml_models = load_ml_models()

# Initialize session state for user management
if 'users' not in st.session_state:
    st.session_state.users = load_user_data()

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'show_login' not in st.session_state:
    st.session_state.show_login = True

if 'show_register' not in st.session_state:
    st.session_state.show_register = False

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

if 'analysis_source' not in st.session_state:
    st.session_state.analysis_source = None

# Utility to hash passwords
def hash_password(password):
    """Create a hashed password"""
    return hashlib.sha256(password.encode()).hexdigest()

# User management functions
def register_user(username, password):
    """Register a new user"""
    if username in st.session_state.users:
        return False, "User already exists"
    
    # Hash the password for security
    hashed_password = hash_password(password)
    
    st.session_state.users[username] = {
        "password": hashed_password,
        "tiktok_username": None,
        "analysis_results": None,
        "video_analysis_results": None,
        "videos": [],
        "created_at": datetime.now().isoformat()
    }
    
    # Save user data to persistent storage
    save_user_data()
    
    return True, "User registered successfully"

def login_user(username, password):
    """Login a user"""
    if username not in st.session_state.users:
        return False, "User not found"
    
    hashed_password = hash_password(password)
    if st.session_state.users[username]["password"] != hashed_password:
        return False, "Invalid password"
    
    st.session_state.current_user = username
    st.session_state.show_login = False
    return True, "Login successful"

def logout_user():
    """Logout the current user"""
    st.session_state.current_user = None
    st.session_state.show_login = True
    st.session_state.show_register = False

def connect_tiktok(username, tiktok_username):
    """Connect a TikTok account to the user"""
    if username not in st.session_state.users:
        return False, "User not found"
    
    st.session_state.users[username]["tiktok_username"] = tiktok_username
    save_user_data()
    return True, "TikTok account connected successfully"

# Video and analysis functions
def analyze_tiktok_feed(username):
    """Analyze user's TikTok feed"""
    user_data = st.session_state.users.get(username, {})
    tiktok_username = user_data.get("tiktok_username")
    
    if not tiktok_username:
        return False, "TikTok account not connected", None
    
    try:
        # 1. Get TikTok feed data - limit to 10 videos
        feed_data = tiktok_api.get_user_feed(tiktok_username, limit=10)
        
        # 2. Process each video
        processed_videos = []
        for video in feed_data:
            processed_video = process_video(video)
            processed_videos.append(processed_video)
        
        # 3. Calculate metrics
        metrics = calculate_metrics(processed_videos)
        
        # 4. Save results
        st.session_state.users[username]["analysis_results"] = metrics
        st.session_state.users[username]["last_analysis"] = datetime.now().isoformat()
        save_user_data()
        
        return True, "Analysis complete", metrics
    except Exception as e:
        log_error(f"Error analyzing TikTok feed: {str(e)}", e)
        return False, f"Analysis failed: {str(e)}", None

def analyze_uploaded_videos(username):
    """Analyze user's uploaded videos"""
    user_data = st.session_state.users.get(username, {})
    videos = user_data.get("videos", [])
    
    # Print debugging information
    print(f"Starting analysis for user: {username}")
    print(f"Found {len(videos)} videos to analyze")
    
    if not videos:
        return False, "No videos available for analysis", None
    
    try:
        # Extract all transcripts
        transcripts = [video['transcript'] for video in videos]
        combined_text = " ".join(transcripts)
        
        print(f"Combined transcript length: {len(combined_text)} characters")
        
        # Process the combined transcript
        is_political = classify_political(combined_text, ml_models)
        print(f"Political content detected: {is_political}")
        
        if not is_political:
            results = {
                "political_percentage": 0,
                "bias_distribution": {"left": 0, "center": 100, "right": 0},
                "echo_chamber_risk": {"score": 0, "category": "Low"},
                "diversity_score": 100,
                "topic_distribution": {"General": 100},
                "recommendations": ["Your videos do not contain significant political content."]
            }
        else:
            # Analyze political bias and topics
            bias = classify_bias(combined_text, ml_models)
            topics = identify_topics(combined_text)
            
            print(f"Detected bias: {bias}")
            print(f"Detected topics: {topics}")
            
            # Calculate bias distribution
            bias_distribution = {"left": 0, "center": 0, "right": 0}
            bias_distribution[bias] = 100
            
            # Calculate topic distribution
            topic_distribution = {}
            for topic in topics:
                topic_distribution[topic] = 100 // len(topics)
            
            # Echo chamber risk based on bias polarization
            risk_score = 80 if bias in ["left", "right"] else 30
            risk_category = "High" if risk_score > 60 else "Moderate" if risk_score > 30 else "Low"
            
            # Generate recommendations
            recommendations = []
            if bias in ["left", "right"]:
                opposite = "right" if bias == "left" else "left"
                recommendations.append(f"Your videos show a strong {bias} bias. Consider exploring {opposite}-leaning perspectives.")
            
            if len(topics) < 3:
                recommendations.append("Your content lacks topic diversity. Try exploring a wider range of political topics.")
            
            results = {
                "political_percentage": 100 if is_political else 0,
                "bias_distribution": bias_distribution,
                "echo_chamber_risk": {"score": risk_score, "category": risk_category},
                "diversity_score": 100 - risk_score,
                "topic_distribution": topic_distribution,
                "recommendations": recommendations
            }
        
        # Print the results for debugging
        print("Analysis results:", results)
        
        # Save the results
        st.session_state.users[username]["video_analysis_results"] = results
        st.session_state.users[username]["last_video_analysis"] = datetime.now().isoformat()
        save_user_data()
        
        print("Results saved to session state and user data")
        
        # Add a direct display of results on the current page for testing
        st.success("Analysis complete!")
        st.write("Debug - Analysis Results:")
        st.json(results)
        
        # Make sure the show_results flag is set
        st.session_state.show_results = True
        st.session_state.analysis_source = "videos"
        
        return True, "Video analysis complete", results
    except Exception as e:
        error_msg = f"Error analyzing videos: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())  # Print full traceback
        return False, error_msg, None
    
def process_video(video):
    """Process a single video using the ML models"""
    text = video.get("text", "")
    
    # Determine if the video is political
    is_political = classify_political(text, ml_models)
    
    if not is_political:
        return {
            "video_id": video["video_id"],
            "text": text,
            "is_political": False,
            "engagement_metrics": {
                "views": video.get("views", 0),
                "likes": video.get("likes", 0),
                "comments": video.get("comments", 0),
                "shares": video.get("shares", 0)
            },
            "engagement_weight": 1.0,
            "create_time": video.get("create_time", 0)
        }
    
    # For political content, determine bias and topics
    bias = classify_bias(text, ml_models)
    topics = identify_topics(text)
    
    # Calculate engagement weight
    engagement_weight = 1.0
    view_count = video.get("views", 0)
    
    if view_count > 0:
        like_ratio = video.get("likes", 0) / view_count
        comment_ratio = video.get("comments", 0) / view_count
        share_ratio = video.get("shares", 0) / view_count
        
        engagement_weight += (like_ratio * 2) + (comment_ratio * 3) + (share_ratio * 4)
    
    return {
        "video_id": video["video_id"],
        "text": text,
        "is_political": True,
        "political_bias": bias,
        "topics": topics,
        "engagement_metrics": {
            "views": video.get("views", 0),
            "likes": video.get("likes", 0),
            "comments": video.get("comments", 0),
            "shares": video.get("shares", 0)
        },
        "engagement_weight": engagement_weight,
        "create_time": video.get("create_time", 0)
    }

def calculate_metrics(processed_videos):
    """Calculate metrics based on processed videos"""
    
    # 1. Calculate political percentage
    total_videos = len(processed_videos)
    political_videos = [v for v in processed_videos if v["is_political"]]
    political_count = len(political_videos)
    political_percentage = int((political_count / total_videos) * 100) if total_videos > 0 else 0
    
    # If no political videos, return minimal results
    if political_count == 0:
        return {
            "political_percentage": 0,
            "bias_distribution": {"left": 0, "center": 100, "right": 0},
            "echo_chamber_risk": {"score": 0, "category": "Low"},
            "diversity_score": 100,
            "topic_distribution": {"General": 100},
            "recommendations": ["Your feed does not contain significant political content."]
        }
    
    # 2. Calculate bias distribution
    bias_counts = {"left": 0, "center": 0, "right": 0}
    total_weight = 0
    
    for video in political_videos:
        bias = video.get("political_bias", "center")
        weight = video.get("engagement_weight", 1.0)
        
        bias_counts[bias] += weight
        total_weight += weight
    
    bias_distribution = {}
    for bias, count in bias_counts.items():
        bias_distribution[bias] = int((count / total_weight) * 100) if total_weight > 0 else 0
    
    # 3. Calculate echo chamber risk
    max_bias = max(bias_distribution.values())
    
    if max_bias > 60:
        risk_score = 80  # High risk
        risk_category = "High"
    elif max_bias > 40:
        risk_score = 50  # Moderate risk
        risk_category = "Moderate"
    else:
        risk_score = 20  # Low risk
        risk_category = "Low"
    
    # 4. Calculate topic distribution
    topic_counts = {}
    
    for video in political_videos:
        for topic in video.get("topics", []):
            weight = video.get("engagement_weight", 1.0)
            
            if topic not in topic_counts:
                topic_counts[topic] = 0
            
            topic_counts[topic] += weight
    
    # Normalize to percentages
    topic_distribution = {}
    for topic, count in topic_counts.items():
        topic_distribution[topic] = int((count / total_weight) * 100) if total_weight > 0 else 0
    
    # Ensure we have at least some topics
    if not topic_distribution:
        topic_distribution = {
            "General_Politics": 100
        }
    
    # 5. Calculate diversity score
    diversity_score = 100 - risk_score
    
    # 6. Generate recommendations
    recommendations = []
    
    # Recommendation based on bias balance
    if max_bias > 60:
        dominant_bias = next(bias for bias, value in bias_distribution.items() if value == max_bias)
        underrepresented_bias = next(bias for bias, value in bias_distribution.items() if value == min(bias_distribution.values()))
        
        recommendations.append(f"Your feed shows mostly {dominant_bias}-leaning content. Try exploring some {underrepresented_bias} perspectives.")
    
    # Recommendation based on topic diversity
    if len(topic_distribution) < 3:
        recommendations.append("Your feed lacks topic diversity. Try exploring a wider range of political topics.")
    elif topic_distribution:
        min_topic = min(topic_distribution.items(), key=lambda x: x[1])
        recommendations.append(f"Your feed lacks content about {min_topic[0].replace('_', ' ')} - explore this topic for a more balanced perspective.")
    
    # General recommendation
    if risk_score > 40:
        recommendations.append("Engage with more diverse content to reduce your echo chamber effect.")
    
    return {
        "political_percentage": political_percentage,
        "bias_distribution": bias_distribution,
        "echo_chamber_risk": {
            "score": risk_score,
            "category": risk_category
        },
        "diversity_score": diversity_score,
        "topic_distribution": topic_distribution,
        "recommendations": recommendations
    }

# UI Functions
def show_login_page():
    st.markdown("<h1 class='main-header'>Filter Bubble Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p>Discover how diverse your social media content truly is</p>", unsafe_allow_html=True)
    
    # Create tabs for login/register
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Login", use_container_width=True)
            
            if login_submit:
                if username and password:
                    with st.spinner("Logging in..."):
                        success, message = login_user(username, password)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.error("Please enter both username and password")
    
    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Username", key="register_username")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            register_submit = st.form_submit_button("Register", use_container_width=True)
            
            if register_submit:
                if new_username and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        with st.spinner("Creating account..."):
                            success, message = register_user(new_username, new_password)
                            if success:
                                st.success(message)
                                st.info("Please log in with your new account")
                            else:
                                st.error(message)
                else:
                    st.error("Please fill in all fields")

def show_main_app():
    username = st.session_state.current_user
    user_data = st.session_state.users.get(username, {})
    
    # Header with logout button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1 class='main-header'>Filter Bubble Analyzer</h1>", unsafe_allow_html=True)
    with col2:
        st.button("Logout", on_click=logout_user, type="primary")
    
    st.markdown(f"<p>Welcome, <strong>{username}</strong>!</p>", unsafe_allow_html=True)
    
    # Main content tabs
    tiktok_tab, upload_tab, results_tab = st.tabs(["TikTok Analysis", "Upload Videos", "Results"])
    
# TikTok Analysis Tab
    with tiktok_tab:
        st.markdown("<h2 class='subheader'>Connect to TikTok</h2>", unsafe_allow_html=True)
        
        # Check if already connected
        tiktok_username = user_data.get("tiktok_username")
        if tiktok_username:
            st.success(f"✅ Connected to TikTok account: **{tiktok_username}**")
            
            # Add option to change account
            if st.button("Change TikTok Account", key="change_tiktok"):
                # Reset TikTok connection
                st.session_state.users[username]["tiktok_username"] = None
                save_user_data()
                st.success("TikTok account disconnected. You can now connect a different account.")
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Analyze Your TikTok Feed")
            st.info("This will analyze your recent TikTok videos for political content and bias. The analysis helps you understand your content consumption patterns and potential filter bubbles.")
            
            # Add clear analyze button
            if st.button("🔍 Analyze TikTok Feed", key="analyze_feed_btn", use_container_width=True):
                analysis_placeholder = st.empty()
                
                with analysis_placeholder.container():
                    with st.spinner("Analyzing your TikTok feed..."):
                        # Create a progress bar for better UX
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Fetching videos
                        status_text.info("Fetching your TikTok videos...")
                        progress_bar.progress(25)
                        time.sleep(0.5)  # Simulate processing time
                        
                        # Step 2: Processing content
                        status_text.info("Processing video content...")
                        progress_bar.progress(50)
                        time.sleep(0.5)  # Simulate processing time
                        
                        # Step 3: Analyzing patterns
                        status_text.info("Analyzing content patterns...")
                        progress_bar.progress(75)
                        time.sleep(0.5)  # Simulate processing time
                        
                        # Final step: Generating insights
                        status_text.info("Generating insights...")
                        progress_bar.progress(90)
                        
                        # Actual analysis
                        success, message, results = analyze_tiktok_feed(username)
                        progress_bar.progress(100)
                        
                        if success:
                            st.session_state.analysis_source = "tiktok"
                            st.session_state.show_results = True
                            analysis_placeholder.success("✅ " + message)
                            time.sleep(1)  # Brief delay before redirecting
                            st.rerun()
                        else:
                            analysis_placeholder.error("❌ " + message)
        else:
            st.info("Connect your TikTok account to analyze your feed content.")
            
            # Create a card-like container for the form
            with st.container():
                st.markdown("""
                <div style="background-color: #2D3748; border-radius: 10px; padding: 20px; border: 1px solid #38425E;">
                    <h3 style="margin-top: 0;">Enter your TikTok username</h3>
                    <p>This will be used to fetch and analyze your most recent videos.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a cleaner form with better spacing
                with st.form("connect_tiktok_form"):
                    st.write("")  # Add some spacing
                    tiktok_username = st.text_input(
                        "TikTok Username",
                        help="Enter your TikTok username without the @ symbol"
                    )
                    
                    st.write("")  # Add some spacing
                    
                    # Add columns for better button placement
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        connect_submit = st.form_submit_button("Connect", use_container_width=True)
                    
                    if connect_submit:
                        if tiktok_username:
                            with st.spinner("Connecting to TikTok..."):
                                success, message = connect_tiktok(username, tiktok_username)
                                if success:
                                    st.success("✅ " + message)
                                    st.rerun()
                                else:
                                    st.error("❌ " + message)
                        else:
                            st.error("Please enter your TikTok username")
            
            # Add some helpful information
            with st.expander("ℹ️ How this works"):
                st.markdown("""
                1. Enter your TikTok username to connect your account
                2. The app will fetch your most recent videos (up to 10)
                3. The content will be analyzed for political bias and topics
                4. You'll receive personalized insights about your content consumption
                
                **Note:** This app doesn't store your TikTok credentials or access private content.
                It only analyzes publicly available videos from your feed.
                """)
    
    
    # Upload Videos Tab
    with upload_tab:
        st.markdown("<h2 class='subheader'>Upload Videos</h2>", unsafe_allow_html=True)
        
        # Display counter
        videos = user_data.get("videos", [])
        videos_count = len(videos)
        max_videos = 10
        
        # Show progress bar with clearer label
        st.write(f"**Videos: {videos_count}/{max_videos}**")
        st.progress(min(videos_count / max_videos, 1.0))
        
        # FFmpeg warning if not available
        ffmpeg_available = is_ffmpeg_available()
        if not ffmpeg_available:
            st.warning("⚠️ FFmpeg is not installed or not found in PATH. Video transcription will be limited. See the sidebar for installation instructions.")
        
        # Check for API key
        st.info("Using local Whisper model for transcription. The first run will download the model, which might take some time depending on your internet connection and model size.")
        
        # Upload section
        if videos_count < max_videos:
            st.info("Upload videos to analyze their content for political bias and topics.")
            
            # Create a unique key for each session to avoid state issues
            if 'upload_key' not in st.session_state:
                st.session_state.upload_key = f"upload_{int(time.time())}"
            
            # File uploader with clear instructions
            uploaded_file = st.file_uploader(
                "Choose a video file (MP4, MOV, AVI, MKV, WEBM)",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                key=st.session_state.upload_key,
                help="Maximum file size: 200MB"
            )
            
            # When a file is selected, show a process button
            if uploaded_file is not None:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"Selected file: **{uploaded_file.name}** ({round(uploaded_file.size/1024/1024, 2)} MB)")
                with col2:
                    # Use a button to trigger processing
                    process_clicked = st.button("Process Video", key="process_video_btn", help="Click to process this video")
                    
                    if process_clicked:
                        # Create progress container
                        progress_container = st.empty()
                        status_text = st.empty()
                        
                        with progress_container.container():
                            progress_bar = st.progress(0)
                        
                        try:
                            # Step 1: Save file (10%)
                            status_text.info("Saving video file...")
                            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            progress_bar.progress(10)
                            
                            # Step 2: Process video (30%)
                            status_text.info("Processing video...")
                            progress_bar.progress(30)
                            
                            # Step 3: Transcribe with Whisper API (80%)
                            status_text.info("Transcribing audio content...")
                            transcript = transcribe_video(file_path)
                            progress_bar.progress(80)
                            
                            # Step 4: Finalize and store (100%)
                            status_text.info("Finalizing...")
                            
                            # Add to user's videos
                            if "videos" not in st.session_state.users[username]:
                                st.session_state.users[username]["videos"] = []
                            
                            # Check if this file already exists
                            existing_file = False
                            for video in st.session_state.users[username]["videos"]:
                                if video['filename'] == uploaded_file.name:
                                    existing_file = True
                                    break
                            
                            if existing_file:
                                status_text.warning(f"A video named '{uploaded_file.name}' already exists in your collection.")
                                progress_container.empty()
                            else:
                                video_info = {
                                    'filename': uploaded_file.name,
                                    'path': file_path,
                                    'transcript': transcript,
                                    'upload_time': time.time()
                                }
                                
                                st.session_state.users[username]["videos"].append(video_info)
                                save_user_data()
                                progress_bar.progress(100)
                                
                                # Clear progress indicators
                                progress_container.empty()
                                status_text.empty()
                                
                                # Success message with transcript preview
                                st.success(f"Video '{uploaded_file.name}' uploaded and transcribed successfully!")
                                
                                with st.expander("View Transcript"):
                                    st.write(transcript)
                                
                                # Generate a new upload key to reset the uploader
                                st.session_state.upload_key = f"upload_{int(time.time())}"
                                
                                # Rerun to refresh the UI
                                st.rerun()
                                
                        except Exception as e:
                            progress_container.empty()
                            status_text.error(f"Error processing video: {str(e)}")
                            log_error("Video processing error", e)
        else:
            st.warning("Maximum number of videos reached (10). Please delete some videos before uploading more.")
        
        # List uploaded videos with clear separation
        if videos:
            st.markdown("---")
            st.markdown("### Your Videos")
            
            for i, video in enumerate(videos):
                with st.expander(f"{i+1}. {video['filename']}"):
                    st.write("**Transcript:**")
                    if len(video['transcript']) > 300:
                        st.write(video['transcript'][:300] + "...")
                        if st.button("Show Full Transcript", key=f"show_transcript_{i}"):
                            st.write(video['transcript'])
                    else:
                        st.write(video['transcript'])
                    
                    # Clear delete button with confirmation
                    if st.button("🗑️ Delete Video", key=f"delete_{i}", help="Delete this video"):
                        if 'delete_confirm' not in st.session_state:
                            st.session_state.delete_confirm = {}
                        
                        st.session_state.delete_confirm[i] = True
                        st.rerun()
                    
                    # Show confirmation dialog
                    if 'delete_confirm' in st.session_state and i in st.session_state.delete_confirm:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("✅ Confirm Delete", key=f"confirm_delete_{i}"):
                                # Remove video file if it exists
                                try:
                                    if os.path.exists(video['path']):
                                        os.remove(video['path'])
                                except Exception as e:
                                    log_error(f"Error removing video file: {str(e)}", e)
                                
                                # Remove from user data
                                st.session_state.users[username]["videos"].pop(i)
                                save_user_data()
                                
                                # Clear confirmation state
                                if 'delete_confirm' in st.session_state:
                                    st.session_state.delete_confirm = {}
                                
                                st.success(f"Video '{video['filename']}' deleted successfully!")
                                st.rerun()
                        with col2:
                            if st.button("❌ Cancel", key=f"cancel_delete_{i}"):
                                # Clear confirmation state
                                if 'delete_confirm' in st.session_state and i in st.session_state.delete_confirm:
                                    del st.session_state.delete_confirm[i]
                                st.rerun()
            
            # Show analyze button at the bottom with more prominence
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Click to analyze all your uploaded videos for political content and bias.")
            with col2:
                if st.button("🔍 Analyze Videos", key="analyze_videos_btn", use_container_width=True):
                    with st.spinner("Analyzing your videos..."):
                        success, message, results = analyze_uploaded_videos(username)
                        if success:
                            st.session_state.analysis_source = "videos"
                            st.session_state.show_results = True
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("No videos uploaded yet. Upload some videos to analyze them.")

# Results Tab
    with results_tab:
        st.markdown("<h2 class='subheader'>Analysis Results</h2>", unsafe_allow_html=True)
        
        # Check if any analysis has been performed
        tiktok_results = user_data.get("analysis_results")
        video_results = user_data.get("video_analysis_results")
        
        if not tiktok_results and not video_results:
            st.info("No analysis results available yet. Use the TikTok Analysis or Upload Videos tabs to analyze content.")
        else:
            # Determine which results to show (most recent or selected)
            selected_source = st.session_state.analysis_source if st.session_state.analysis_source else "videos" if video_results else "tiktok"
            current_results = video_results if selected_source == "videos" else tiktok_results
            
            # Show source selection if both analyses are available
            if tiktok_results and video_results:
                source_options = {"tiktok": "TikTok Feed Analysis", "videos": "Uploaded Videos Analysis"}
                selected_source = st.selectbox(
                    "Select analysis source:", 
                    options=list(source_options.keys()),
                    format_func=lambda x: source_options[x],
                    index=0 if selected_source == "tiktok" else 1
                )
                current_results = tiktok_results if selected_source == "tiktok" else video_results
                st.session_state.analysis_source = selected_source
            
            # Show last analysis time
            if selected_source == "tiktok" and "last_analysis" in user_data:
                analysis_time = datetime.fromisoformat(user_data["last_analysis"])
                st.caption(f"TikTok analysis performed on {analysis_time.strftime('%B %d, %Y at %H:%M')}")
            elif selected_source == "videos" and "last_video_analysis" in user_data:
                analysis_time = datetime.fromisoformat(user_data["last_video_analysis"])
                st.caption(f"Video analysis performed on {analysis_time.strftime('%B %d, %Y at %H:%M')}")
            
            # Display results in a dashboard layout
            if current_results:
                st.markdown("### Key Insights")
                
                # Create metrics in a 3-column layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    political_percentage = current_results.get("political_percentage", 0)
                    st.markdown(f"""
                    <div class="card">
                        <p>Political Content</p>
                        <p class="metric-value">{political_percentage}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    risk = current_results.get("echo_chamber_risk", {})
                    risk_score = risk.get("score", 0)
                    risk_category = risk.get("category", "Low")
                    risk_class = "risk-low" if risk_category == "Low" else "risk-moderate" if risk_category == "Moderate" else "risk-high"
                    
                    st.markdown(f"""
                    <div class="card">
                        <p>Echo Chamber Risk</p>
                        <p class="metric-value {risk_class}">{risk_category}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    diversity_score = current_results.get("diversity_score", 100)
                    diversity_class = "risk-high" if diversity_score > 70 else "risk-moderate" if diversity_score > 40 else "risk-low"
                    
                    st.markdown(f"""
                    <div class="card">
                        <p>Diversity Score</p>
                        <p class="metric-value {diversity_class}">{diversity_score}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add bias distribution chart
                st.markdown("### Political Bias Distribution")
                bias_data = current_results.get("bias_distribution", {"left": 0, "center": 0, "right": 0})
                
                # Create a DataFrame for the chart
                bias_df = pd.DataFrame({
                    'Bias': list(bias_data.keys()),
                    'Percentage': list(bias_data.values())
                })
                
                # Create colors for the chart
                bias_colors = {
                    'left': '#3366CC',    # Blue for left
                    'center': '#9467BD',  # Purple for center
                    'right': '#CC3333'    # Red for right
                }
                
                # Create a bar chart
                bias_chart = alt.Chart(bias_df).mark_bar().encode(
                    x=alt.X('Bias:N', title='Political Bias', sort=['left', 'center', 'right']),
                    y=alt.Y('Percentage:Q', title='Percentage (%)'),
                    color=alt.Color('Bias:N', scale=alt.Scale(
                        domain=list(bias_colors.keys()),
                        range=list(bias_colors.values())
                    )),
                    tooltip=['Bias:N', 'Percentage:Q']
                ).properties(
                    height=300
                )
                
                st.altair_chart(bias_chart, use_container_width=True)


                # Add topic distribution
                st.markdown("### Topic Distribution")
                topic_data = current_results.get("topic_distribution", {"General": 100})

                # Create DataFrame for topics
                topic_df = pd.DataFrame({
                    'Topic': list(topic_data.keys()),
                    'Percentage': list(topic_data.values())
                })

                # Sort by percentage value (descending)
                topic_df = topic_df.sort_values(by='Percentage', ascending=False)

                # Improve topic labels for better readability
                topic_df['Topic'] = topic_df['Topic'].apply(lambda x: x.replace('_', ' '))

                # Create a base chart with configuration
                base = alt.Chart(topic_df).encode(
                    y=alt.Y('Topic:N', 
                            title=None, 
                            sort='-x',
                            axis=alt.Axis(labelFontSize=14, labelFontWeight='bold')),
                    x=alt.X('Percentage:Q', 
                            title='Percentage (%)',
                            scale=alt.Scale(domain=[0, 100]),
                            axis=alt.Axis(grid=True, tickMinStep=10))
                ).properties(
                    height=40 * len(topic_data),
                    width=600
                )

                # Define the bar chart
                bars = base.mark_bar().encode(
                    color=alt.Color('Percentage:Q', 
                                scale=alt.Scale(scheme='blues', domain=[0, 100]),
                                legend=alt.Legend(title="Percentage")),
                    tooltip=[
                        alt.Tooltip('Topic:N', title='Topic'),
                        alt.Tooltip('Percentage:Q', title='Percentage', format='.1f')
                    ]
                )

                # Define the text labels
                text = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=5,  # Offset from end of bar
                    color='white',
                    fontSize=14,
                    fontWeight='bold'
                ).encode(
                    text=alt.Text('Percentage:Q', format='.0f%')
                )

                # Layer the charts with configuration at the LayerChart level
                final_chart = alt.layer(bars, text).configure_axis(
                    grid=True,
                    gridColor='#444444',
                    labelColor='white',
                    titleColor='white'
                ).configure_view(
                    strokeWidth=0
                )

                st.altair_chart(final_chart, use_container_width=True)

                # Optional: Add pie chart option for alternative visualization
                with st.expander("Alternative View: Pie Chart"):
                    # Create a pie chart using matplotlib
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Custom color map that works well with dark theme
                    colors = plt.cm.Blues(np.linspace(0.6, 1.0, len(topic_df)))
                    
                    # Create pie chart
                    wedges, texts, autotexts = ax.pie(
                        topic_df['Percentage'], 
                        labels=topic_df['Topic'],
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors,
                        wedgeprops={'edgecolor': '#111827', 'linewidth': 1.5}
                    )
                    
                    # Styling the text
                    for text in texts:
                        text.set_color('white')
                        text.set_fontsize(12)
                        text.set_fontweight('bold')
                    
                    for autotext in autotexts:
                        autotext.set_color('#111827')
                        autotext.set_fontsize(10)
                        autotext.set_fontweight('bold')
                    
                    # Equal aspect ratio ensures that pie is drawn as a circle
                    ax.axis('equal')
                    
                    # Set background color to match the app's dark theme
                    fig.patch.set_facecolor('#111827')
                    ax.set_facecolor('#111827')
                    
                    plt.title('Topic Distribution', color='white', fontsize=16, pad=20)
                    
                    st.pyplot(fig)

                # Optional: Add a table view for exact numbers
                with st.expander("View Data Table"):
                    st.dataframe(
                        topic_df.set_index('Topic'),
                        use_container_width=True,
                        column_config={"Percentage": st.column_config.ProgressColumn(
                            "Percentage",
                            format="%d%%",
                            min_value=0,
                            max_value=100,
                        )}
                    )
                                
                
                # Recommendations section
                st.markdown("### Personalized Recommendations")
                recommendations = current_results.get("recommendations", [])
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="recommendation">
                            {rec}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific recommendations available.")
                
                # Action buttons
                st.markdown("### Next Steps")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Export Results (PDF)", key="export_pdf_btn"):
                        st.info("Export functionality would be implemented here.")
                
                with col2:
                    if st.button("🔄 Re-analyze Content", key="reanalyze_btn"):
                        if selected_source == "tiktok":
                            st.session_state.show_results = False
                            st.session_state.analysis_source = None
                            st.rerun()
                        else:
                            st.session_state.show_results = False
                            st.session_state.analysis_source = None
                            st.rerun()

# Main app logic
def main():
    # Check if ffmpeg is available
    ffmpeg_available = is_ffmpeg_available()
    
    # Check if user is logged in
    if st.session_state.current_user:
        show_main_app()
    else:
        show_login_page()

if __name__ == "__main__":
    main()