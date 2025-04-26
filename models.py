import os
import re
import pickle
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Path to model files
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class TextPreprocessor:
    """Text preprocessing for political content analysis"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text):
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        # Rejoin
        return " ".join(tokens)

def train_political_classifier():
    """Train a model to classify political vs. non-political content"""
    # This would typically load your labeled dataset
    # For this example, we'll create a simple synthetic dataset
    political_texts = [
        "The government should increase spending on infrastructure",
        "Congress needs to pass legislation to address climate change",
        "The president's economic policies have been controversial",
        "Voters are concerned about healthcare and taxes",
        "The Supreme Court's ruling on civil rights was significant",
        "Campaign finance reform is needed for fair elections",
        "The senator introduced a bill to protect voting rights",
        "Foreign policy decisions impact our standing in the world",
        "The debate over gun control continues in state legislatures",
        "Immigration reform remains a divisive political issue"
    ]
    
    non_political_texts = [
        "This recipe for chocolate cake is delicious",
        "My dog loves to play fetch in the park",
        "The movie was exciting from start to finish",
        "I bought a new phone and it works great",
        "The concert last night had amazing sound quality",
        "The hotel room had a beautiful ocean view",
        "My garden is blooming with colorful flowers",
        "This book is a fascinating science fiction story",
        "The football game went into overtime",
        "I learned how to play the guitar this summer"
    ]
    
    # Create labeled dataset
    X = political_texts + non_political_texts
    y = [True] * len(political_texts) + [False] * len(non_political_texts)
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    X_processed = [preprocessor.preprocess(text) for text in X]
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X_processed)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_vectorized, y)
    
    # Save model and vectorizer
    joblib.dump(model, f'{MODEL_DIR}/political_classifier.joblib')
    joblib.dump(vectorizer, f'{MODEL_DIR}/political_vectorizer.joblib')
    
    return model, vectorizer

def train_bias_classifier():
    """Train a model to classify political bias (left, center, right)"""
    # Synthetic dataset
    left_texts = [
        "We need more social programs to help the disadvantaged",
        "Universal healthcare is a human right that should be guaranteed",
        "Income inequality must be addressed through progressive taxation",
        "Climate change requires immediate government intervention",
        "Systemic racism needs to be dismantled through policy reform"
    ]
    
    center_texts = [
        "Both parties need to compromise to solve our problems",
        "Practical solutions matter more than ideological purity",
        "We need to find middle ground on controversial issues",
        "Balanced approaches that consider multiple viewpoints are best",
        "Nonpartisan cooperation is essential for effective governance"
    ]
    
    right_texts = [
        "Lower taxes stimulate economic growth and job creation",
        "Government regulations stifle innovation and business",
        "Strong border security is essential for national safety",
        "Individual liberty should be prioritized over government mandates",
        "Traditional values provide important social stability"
    ]
    
    # Create labeled dataset
    X = left_texts + center_texts + right_texts
    y = ["left"] * len(left_texts) + ["center"] * len(center_texts) + ["right"] * len(right_texts)
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    X_processed = [preprocessor.preprocess(text) for text in X]
    
    # Vectorize (use n-grams for better bias detection)
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
    X_vectorized = vectorizer.fit_transform(X_processed)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_vectorized, y)
    
    # Save model and vectorizer
    joblib.dump(model, f'{MODEL_DIR}/bias_classifier.joblib')
    joblib.dump(vectorizer, f'{MODEL_DIR}/bias_vectorizer.joblib')
    
    return model, vectorizer

def train_topic_classifier():
    """Train a model to classify political topics"""
    # This would be a multi-label classification problem
    # For simplicity, we'll just create a placeholder for the function
    # In production, you would train a real model on a labeled dataset
    
    # Placeholder to indicate this function exists
    return None, None

def load_models():
    """Load trained models or train new ones if they don't exist"""
    models = {}
    preprocessor = TextPreprocessor()
    models['preprocessor'] = preprocessor
    
    # Political classifier
    political_model_path = f'{MODEL_DIR}/political_classifier.joblib'
    political_vec_path = f'{MODEL_DIR}/political_vectorizer.joblib'
    
    if os.path.exists(political_model_path) and os.path.exists(political_vec_path):
        models['political_classifier'] = joblib.load(political_model_path)
        models['political_vectorizer'] = joblib.load(political_vec_path)
        print("Loaded political classifier model")
    else:
        print("Training political classifier model")
        models['political_classifier'], models['political_vectorizer'] = train_political_classifier()
    
    # Bias classifier
    bias_model_path = f'{MODEL_DIR}/bias_classifier.joblib'
    bias_vec_path = f'{MODEL_DIR}/bias_vectorizer.joblib'
    
    if os.path.exists(bias_model_path) and os.path.exists(bias_vec_path):
        models['bias_classifier'] = joblib.load(bias_model_path)
        models['bias_vectorizer'] = joblib.load(bias_vec_path)
        print("Loaded bias classifier model")
    else:
        print("Training bias classifier model")
        models['bias_classifier'], models['bias_vectorizer'] = train_bias_classifier()
    
    return models

def classify_political(text, models):
    """Determine if content is political"""
    if not text or len(text.strip()) < 10:
        return False
    
    # Use ML model if available
    try:
        preprocessor = models['preprocessor']
        vectorizer = models['political_vectorizer']
        classifier = models['political_classifier']
        
        processed_text = preprocessor.preprocess(text)
        features = vectorizer.transform([processed_text])
        
        return bool(classifier.predict(features)[0])
    except (KeyError, Exception) as e:
        print(f"Error in political classification: {str(e)}")
        # Fallback to keyword matching
        political_keywords = [
            'politics', 'election', 'government', 'democrat', 'republican',
            'policy', 'vote', 'congress', 'president', 'senator', 'representative',
            'legislation', 'democracy', 'constitution', 'amendment', 'supreme court',
            'bill', 'law', 'regulation', 'liberal', 'conservative', 'progressive'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for keyword in political_keywords if keyword in text_lower)
        return matches >= 2

def classify_bias(text, models):
    """Classify political bias"""
    if not text or len(text.strip()) < 10:
        return "center"
    
    # Use ML model if available
    try:
        preprocessor = models['preprocessor']
        vectorizer = models['bias_vectorizer']
        classifier = models['bias_classifier']
        
        processed_text = preprocessor.preprocess(text)
        features = vectorizer.transform([processed_text])
        
        return classifier.predict(features)[0]
    except (KeyError, Exception) as e:
        print(f"Error in bias classification: {str(e)}")
        # Fallback to keyword matching
        left_keywords = [
            'progressive', 'liberal', 'democrat', 'socialism', 'equality',
            'diversity', 'inclusion', 'regulation', 'government program',
            'social justice', 'workers rights', 'universal healthcare'
        ]
        
        right_keywords = [
            'conservative', 'republican', 'freedom', 'liberty', 'tradition',
            'family values', 'deregulation', 'small government', 'free market',
            'second amendment', 'pro-life', 'border security'
        ]
        
        centrist_keywords = [
            'bipartisan', 'moderate', 'compromise', 'common ground', 'pragmatic',
            'practical solution', 'middle path', 'balanced approach', 'both sides'
        ]
        
        text_lower = text.lower()
        
        left_count = sum(keyword in text_lower for keyword in left_keywords)
        right_count = sum(keyword in text_lower for keyword in right_keywords)
        center_count = sum(keyword in text_lower for keyword in centrist_keywords) * 1.5
        
        if left_count > right_count and left_count > center_count:
            return "left"
        elif right_count > left_count and right_count > center_count:
            return "right"
        else:
            return "center"

def identify_topics(text):
    """Identify political topics in text"""
    if not text or len(text.strip()) < 10:
        return ["General_Politics"]
    
    # This would ideally use a multi-label classifier
    # For now, we'll use a more sophisticated keyword approach
    
    topic_keywords = {
        "Elections": ["election", "vote", "ballot", "voter", "campaign", "candidate", 
                     "polling", "primary", "caucus", "electoral", "gerrymandering", "district"],
        "Political_Figures": ["president", "senator", "governor", "congressman", "representative",
                             "candidate", "politician", "leader", "minister", "secretary", "mayor",
                             "administration", "cabinet", "official"],
        "Social_Issues": ["abortion", "gun", "immigration", "race", "rights", "equality", 
                         "discrimination", "protests", "justice", "reform", "welfare", "poverty",
                         "education", "marijuana", "legalization", "police", "criminal", "prison"],
        "Economic_Policy": ["economy", "tax", "budget", "spending", "fiscal", "debt", "deficit",
                           "stimulus", "subsidy", "tariff", "trade", "wage", "income", "inflation",
                           "unemployment", "regulation", "deregulation", "business", "corporation"],
        "International_Relations": ["foreign", "international", "war", "peace", "diplomacy", 
                                   "treaty", "sanctions", "alliance", "military", "defense", "security",
                                   "terrorism", "refugee", "immigration", "border", "global"],
        "Healthcare": ["healthcare", "medical", "insurance", "patient", "doctor", "hospital", 
                      "medicare", "medicaid", "affordable", "prescription", "drug", "public option",
                      "single payer", "universal", "pandemic", "vaccine", "mandate"],
        "Environment": ["climate", "environment", "green", "pollution", "energy", "renewable",
                       "sustainable", "emission", "carbon", "fossil", "conservation", "EPA",
                       "regulation", "paris", "agreement", "warming", "wildlife", "forest"],
        "Constitutional_Issues": ["constitution", "amendment", "rights", "supreme court", "judicial",
                                "freedom", "speech", "religion", "press", "assembly", "privacy",
                                "due process", "civil liberties", "separation", "powers"]
    }
    
    # Create a frequency map for each topic
    text_lower = text.lower()
    topic_scores = {}
    
    for topic, keywords in topic_keywords.items():
        # Weight more specific keywords higher
        weighted_matches = 0
        for keyword in keywords:
            count = text_lower.count(keyword)
            # Longer keywords are more specific, give them higher weight
            keyword_weight = 0.5 + (len(keyword) / 20)  # Scale length to a reasonable weight
            weighted_matches += count * keyword_weight
        
        # Only count topics with significant presence
        if weighted_matches >= 1.5:
            topic_scores[topic] = weighted_matches
    
    # Sort topics by score and take the top ones
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top topics or general politics if none found
    if sorted_topics:
        return [topic for topic, score in sorted_topics[:3]]  # Take up to 3 top topics
    else:
        return ["General_Politics"]

# If the module is run directly, train models
if __name__ == "__main__":
    print("Training and saving models...")
    load_models()
    print("Done!")