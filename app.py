from flask import Flask, render_template, request, session,redirect
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
from keybert import KeyBERT
import re
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__, template_folder='templates')
app.secret_key = "orgpick_secret_123"  


# Load data
df = pd.read_csv("data/product_dataset.csv")

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "how are you", "what's up"]
    text = text.lower().strip()
    return any(greet in text for greet in greetings)

def get_groq_reply(user_input, profile):
    prompt = f"""
You are a friendly and knowledgeable natural health assistant.

If the message is not a health query, just respond like a helpful friend.

    User Profile:
    - Name: {profile.get('name', 'Unknown')}
    - Age: {profile.get('age', 'N/A')}
    - Gender: {profile.get('gender', 'N/A')}
    - Diet: {profile.get('diet', 'N/A')}
    - Goal: {profile.get('goal', 'N/A')}
    - Allergies: {profile.get('allergies', 'None')}
    - Lifestyle: {profile.get('lifestyle', 'Not specified')}

    Instructions:
    - Greet the user using their first name only once (e.g., "Hi Touheed").
    - Avoid excessive enthusiasm (no more than one friendly sentence).
    - Greet the user in a warm and natural tone using their name only once.
    - Avoid being overly enthusiastic or repetitive (e.g., no multiple greetings or exaggerated compliments).
    - If the message is a casual greeting (like "hi" or "hello"), respond kindly and simply ask how you can help.
    - If it's a health-related question, give friendly, calm guidance or natural remedies if asked.
    - Keep the message brief, warm, and useful.
    - If they mention health issues, offer natural advice and encourage them gently.
    - If the user asks for remedies or has symptoms, try to be helpful and suggest home remedies or health tips.
    - If no health issue is mentioned, keep the chat friendly and offer general wellness advice.
    - End replies with a friendly follow-up like "Would you like help with anything else?"

    Now answer this user message: {user_input}
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10  # Optional: Add timeout for safety
        )

        result = response.json()

        # Debugging help
        print("Groq response:", result)

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "Oops! Couldn't understand your request. Can you please rephrase it?"
    except Exception as e:
        print("Groq API Error:", e)
        return "Sorry, I'm having trouble accessing my health advice brain right now. Please try again later!"


# ✅ Normalize function for consistent comparison
def normalize_text(text):
    return re.sub(r"\s+", "", text.lower().strip())

# ✅ Preprocess dataset
df['normalized_problem'] = df['Health Problem'].apply(normalize_text)
df['embedding'] = df['normalized_problem'].apply(lambda x: model.encode(x))

# ✅ Extract keyphrase from user input
def extract_keyphrases(text, num_phrases=3):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=num_phrases)
    phrases = [kw[0] for kw in keywords]
    return phrases[0] if phrases else text

# ✅ External remedy scraping (fallback)
def fetch_external_recommendations(query):
    try:
        search_url = f"https://www.google.com/search?q=natural+remedies+for+{query.replace(' ', '+')}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        snippets = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')
        recommendations = []

        for snippet in snippets:
            text = snippet.get_text().strip()
            if len(text) > 30 and text not in recommendations:
                recommendations.append(text)
            if len(recommendations) >= 3:
                break

        if recommendations:
            response = "Here are some external natural remedy suggestions:<br><br>"
            for rec in recommendations:
                response += f"- {rec}<br>"
            return response
        else:
            return "I'm sorry, I couldn't find any external recommendations either."
    except Exception as e:
        print("Error while fetching external data:", e)
        return "An error occurred while fetching external recommendations."

@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg")
    if not user_input or user_input.strip() == "":
        return "Please describe your health concern so I can help you."

    profile = session.get("profile", {})
    name = profile.get("name", "")

    try:
        llm_response = get_groq_reply(user_input, profile)
    except Exception as e:
        print("Groq API Error:", e)
        llm_response = "I'm here to help with your health or wellness questions!"

    response = ""
   
    # Always show Groq response first
    response += llm_response + "<br><br>"

    # Check if user message is just a greeting
    if is_greeting(user_input):
        return response + "Would you like help with a health concern or natural remedy?"

    # If it's a health query, extract keyphrase and match products
    keyphrase = extract_keyphrases(user_input)
    normalized_keyphrase = normalize_text(keyphrase)
    user_embedding = model.encode(normalized_keyphrase)

    df["similarity"] = df["embedding"].apply(lambda x: util.cos_sim(x, user_embedding).item())
    top_matches = df[df["similarity"] > 0.6].sort_values(by="similarity", ascending=False)

    shown = set()

    if not top_matches.empty:
        response += "<strong>Based on your concern, here are some natural product recommendations:</strong><br><br>"
        for _, row in top_matches.iterrows():
            product = row['Product']
            benefit = row['Health Benefit']
            unique_key = f"{product.lower().strip()}|{benefit.lower().strip()}"

            if unique_key not in shown:
                search_term = product.replace(" ", "+")
                response += f"""<div style="margin-bottom: 12px;">
                    <strong>{product}</strong> – {benefit}<br>
                    🔗 <a href="https://www.amazon.in/s?k={search_term}" target="_blank">Amazon</a> |
                    <a href="https://www.flipkart.com/search?q={search_term}" target="_blank">Flipkart</a> |
                    <a href="https://www.patanjaliayurved.net/search?query={search_term}" target="_blank">Patanjali</a>
                </div>"""
                shown.add(unique_key)
    else:
        # No internal matches, fallback to external search
        response += "<br><strong>Couldn't find a direct match, but here are some external suggestions:</strong><br><br>"
        response += fetch_external_recommendations(user_input)

    return response

    
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        session['profile'] = {
            'name': request.form.get("name"),
            'age': request.form.get("age"),
            'gender': request.form.get("gender"),
            'diet': request.form.get("diet"),
            'goal': request.form.get("goal"),
            'allergies': request.form.get("allergies"),
            'lifestyle': request.form.get("lifestyle"),
        }
        return redirect("/")  # Redirect to chatbot after saving

    # 👉 Pre-fill form with session data if available
    profile_data = session.get("profile", {})
    return render_template("profile.html", profile=session.get("profile", {}))

@app.route("/")
def home():
    if "profile" not in session:
        return redirect("/profile")  # Redirects to the profile form first
    return render_template("index.html")  # If profile is filled, show chatbot

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)    