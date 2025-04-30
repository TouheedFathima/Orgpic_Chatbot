from flask import Flask, render_template, request, session, redirect
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

def get_groq_reply(user_input, profile, conversation_context=None):
    prompt = f"""
You are Piciki, a friendly and knowledgeable natural health assistant specializing in organic remedies.

User Profile:
- Name: {profile.get('name', 'Unknown')}
- Age: {profile.get('age', 'N/A')}
- Gender: {profile.get('gender', 'N/A')}
- Diet: {profile.get('diet', 'N/A')}
- Goal: {profile.get('goal', 'N/A')}
- Allergies: {profile.get('allergies', 'None')}
- Lifestyle: {profile.get('lifestyle', 'Not specified')}

Conversation History:
{conversation_context or 'No prior conversation'}

Instructions:
- Use a warm, natural tone. Greet the user by their first name ONLY in the first response of a new conversation.
- Keep responses concise, readable, and well-structured:
  - Use short paragraphs (1-2 sentences max).
  - Separate distinct ideas (e.g., context, questions, remedies) into different paragraphs.
  - When listing remedies, use a numbered list with clear, actionable steps.
- Avoid repetitive greetings or overly enthusiastic responses.
- If the user input is a greeting (e.g., "hi", "hello", "hlo"), respond kindly, reset the conversation, and ask how you can help (e.g., "Hey [Name], nice to hear from you! What's on your mind?").
- If the user mentions a health problem, acknowledge it and ask at least two relevant clarifying questions to understand their condition better, unless you have enough information to proceed to recommendations earlier.
- Condition-specific question guidance:
  - For "hair fall" or "hairfall": Ask about symptoms (e.g., dandruff, itchiness), lifestyle changes (e.g., stress, diet), or environmental factors (e.g., travel, water quality).
  - For "PCOS": Ask about symptoms (e.g., irregular periods, facial hair) or diet/lifestyle changes.
  - For "acne": Ask about symptoms (e.g., location of acne) or skincare/diet triggers.
  - For other health problems, ask symptom-based or lifestyle-related questions relevant to the condition.
- Use the user’s responses and conversation history to ask follow-up questions that build on prior answers, maintaining context and avoiding repetition.
- Proceed to recommendations when you have enough information (e.g., after 2 or more questions, or if the problem is clear earlier):
  - Suggest organic products or natural remedies tailored to the user’s condition, profile, and conversation history.
  - Present remedies in a numbered list with clear steps (e.g., "1. Massage warm coconut oil into your scalp for 1 hour before shampooing.").
  - Check the provided dataset for specific products (e.g., "Neem oil for hair fall with dandruff") and include clickable purchase links.
  - If no dataset matches are found, provide general advice and indicate that external sources will be checked for additional recommendations.
- Keep responses brief, helpful, and end with a follow-up question or "Would you like help with anything else?" when recommending.
- If the user says "reset", clear the conversation history and start fresh.
- Do not reset the conversation or misinterpret responses as greetings unless they are clearly greetings (e.g., "hi", "hello") or "reset".
- Maintain context from the conversation history to avoid referencing incorrect topics (e.g., don’t mention sleep if the topic is hair fall).
- If the user uses casual inputs like "hey" but continues the same topic, treat it as a continuation, not a reset.

Current User Message: {user_input}
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
            timeout=10
        )
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return "Oops! Couldn't understand your request. Can you please rephrase it?"
    except Exception as e:
        print("Groq API Error:", e)
        return "Sorry, I'm having trouble right now. Please try again later!"

def normalize_text(text):
    return re.sub(r"\s+", "", text.lower().strip())

# Preprocess dataset
df['normalized_problem'] = df['Health Problem'].apply(normalize_text)
df['embedding'] = df['normalized_problem'].apply(lambda x: model.encode(x))

def extract_keyphrases(text, num_phrases=3):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=num_phrases)
    phrases = [kw[0] for kw in keywords]
    return phrases[0] if phrases else text

def fetch_external_recommendations(query):
    try:
        search_url = f"https://www.google.com/search?q=natural+remedies+for+{query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
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
            return "I'm sorry, I couldn't find any external recommendations."
    except Exception as e:
        print("Error while fetching external data:", e)
        return "An error occurred while fetching external recommendations."

@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg")
    if not user_input or user_input.strip() == "":
        return "Please describe your health concern so I can help you."

    profile = session.get("profile", {})
    conversation = session.get("conversation", {
        "state": "initial",
        "health_problem": None,
        "context": [],
        "question_count": 0
    })

    response = ""
    keyphrase = extract_keyphrases(user_input)
    normalized_keyphrase = normalize_text(keyphrase)

    # Handle reset request
    if user_input.lower().strip() == "reset":
        conversation = {"state": "initial", "health_problem": None, "context": [], "question_count": 0}
        session["conversation"] = conversation
        session.modified = True
        print("Conversation reset:", conversation)
        return "Let's start fresh! What's your health concern?"

    # Build conversation context
    conversation_context = "\n".join([f"User: {c['user']}\nBot: {c['bot']}" for c in conversation["context"]])
    print("Current context:", conversation_context)

    # Get LLM response
    llm_response = get_groq_reply(user_input, profile, conversation_context)
    conversation["context"].append({"user": user_input, "bot": llm_response})

    # Detect health problem if initial
    if conversation["state"] == "initial":
        for problem in df["Health Problem"].str.lower().unique():
            if problem in normalized_keyphrase:
                conversation["state"] = "questioning"
                conversation["health_problem"] = problem
                conversation["question_count"] = 1
                break

    # Update question count and state
    if conversation["state"] == "questioning":
        conversation["question_count"] += 1
        # Move to recommendations after 2 questions if enough info or if LLM suggests remedies
        if (conversation["question_count"] >= 2 and any(kw in llm_response.lower() for kw in ["recommend", "suggest", "try", "use"])) or conversation["question_count"] >= 4:
            conversation["state"] = "recommendation"
        else:
            conversation["state"] = "questioning"

    session["conversation"] = conversation
    session.modified = True
    print("Updated conversation:", conversation)

    # Handle recommendation mode
    if conversation["state"] == "recommendation":
        user_embedding = model.encode(normalized_keyphrase)
        df["similarity"] = df["embedding"].apply(lambda x: util.cos_sim(x, user_embedding).item())
        top_matches = df[df["similarity"] > 0.6].sort_values(by="similarity", ascending=False)

        shown = set()
        if not top_matches.empty:
            response += "<strong>Here are some natural product recommendations for your concern:</strong><br><br>"
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
            response += "<br><strong>Couldn't find a direct match in the dataset. Here are some external suggestions:</strong><br><br>"
            response += fetch_external_recommendations(user_input)

        conversation["state"] = "completed"
        session["conversation"] = conversation
        session.modified = True
        response = llm_response + "<br><br>" + response
        print("Recommendations sent:", response)
        return response

    return llm_response

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
        session.modified = True
        return redirect("/")

    profile_data = session.get("profile", {})
    return render_template("profile.html", profile=profile_data)

@app.route("/")
def home():
    if "profile" not in session:
        return redirect("/profile")
    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port, debug=True)