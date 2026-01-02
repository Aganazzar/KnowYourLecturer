import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# --- Load environment variables from .env ---
load_dotenv()

# --- Get your Groq API key ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# --- Setup Groq client ---
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# --- Load CSV with reviews ---
df = pd.read_csv("reviews.csv")
if "Person_Name" not in df.columns or "Review_Text" not in df.columns or "Rating" not in df.columns:
    raise ValueError("CSV must contain columns: Person_Name, Review_Text, Rating")

# --- Ask for lecturer name ---
lecturer_name = input("Enter the lecturer's name: ").strip()

# Filter reviews for that lecturer
lecturer_reviews = df[df["Person_Name"].str.lower() == lecturer_name.lower()]

if lecturer_reviews.empty:
    print(f"No reviews found for lecturer '{lecturer_name}'.")
    exit()

reviews_list = lecturer_reviews["Review_Text"].tolist()
ratings_list = lecturer_reviews["Rating"].tolist()

# --- Helper functions ---
def build_short_context(person_name, reviews, ratings):
    avg_rating = sum(ratings) / len(ratings)
    review_text = "\n".join([f"- {r}" for r in reviews])
    
    context = f"""
You are an AI that writes short, casual, student-friendly reviews for college lecturers. 
Use simple words, write like a real student would, and keep it short (2–3 sentences). 
Do NOT sound formal, business-like, or robotic.
 
Lecturer: {person_name}

Here are student reviews for this lecturer:
{review_text}

Average rating from students: {avg_rating:.1f}

Task:
1. Summarize all reviews into a concise 2–3 sentence review.
2. Suggest a final rating out of 5 stars based on the text reviews and numeric ratings.
3. Keep the tone academic, helpful, and appropriate for college students.
"""
    return context

def generate_short_review(person_name, reviews, ratings):
    context = build_short_context(person_name, reviews, ratings)
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=context
    )
    return response.output_text

# --- Chunking for large number of reviews ---
def chunk_reviews(reviews, chunk_size=10):
    for i in range(0, len(reviews), chunk_size):
        yield reviews[i:i + chunk_size]

def generate_final_short_review(person_name, reviews, ratings, chunk_size=10):
    mini_summaries = []
    temp_ratings = ratings.copy()

    # Summarize in chunks
    for chunk in chunk_reviews(reviews, chunk_size):
        chunk_ratings = temp_ratings[:len(chunk)]
        temp_ratings = temp_ratings[len(chunk):]
        mini_summary = generate_short_review(person_name, chunk, chunk_ratings)
        mini_summaries.append(mini_summary)

    # Combine mini-summaries into final 2–3 line review
    final_context = f"""
You are an AI that writes short, casual, student-friendly reviews for college lecturers. 
Use simple words, write like a real student would, and keep it short (2–3 sentences). 
Do NOT sound formal, business-like, or robotic.
 

Lecturer: {person_name}

Here are summarized chunks of student reviews:
{chr(10).join(['- ' + s for s in mini_summaries])}

Task:
1. Summarize all the above into a concise 2–3 sentence review.
2. Suggest a final rating out of 5 stars based on the reviews.
3. Keep the tone academic, helpful, and appropriate for college students.
"""
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=final_context
    )
    return response.output_text

# --- Generate review for the requested lecturer ---
final_review_text = generate_final_short_review(
    lecturer_name,
    reviews_list,
    ratings_list,
    chunk_size=10
)

print("\n--- AI-Generated Review ---")
print(final_review_text)
