from flask import Flask, render_template, request, session, redirect
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import os
from keybert import KeyBERT
from dotenv import load_dotenv
import requests
from pymongo import MongoClient
from fuzzywuzzy import fuzz

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__, template_folder='templates')
app.secret_key = "orgpick_secret_123"

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["momy_db"]
product_collection = db["products"]

# Load data from MongoDB into a DataFrame
def load_products():
    products = list(product_collection.find({}, {'_id': 0}))
    return pd.DataFrame(products)

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

def get_groq_reply(user_input, profile, conversation_context=None, dataset_products=None):
    prompt = f"""
You are Piciki, a friendly and knowledgeable natural health assistant for Momy's Farm, specializing in organic remedies.

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

Dataset Products:
{dataset_products or 'No specific products provided'}

Instructions:
- Use a warm, friendly, and concise tone. Greet the user by their first name ONLY in the first response of a new conversation. Do not use the user’s name in subsequent responses unless starting a new session.
- Keep responses short, readable, and well-structured:
  - Use short paragraphs (1-2 sentences).
  - Use numbered lists for remedies or products with clear steps.
- If the user input is a greeting (e.g., "hi", "hello", "hlo"), respond kindly, reset the conversation, and ask how you can help.
- If the user explicitly requests remedies or products (e.g., contains "remedies," "products," "give me," "don’t ask," "just tell me"), provide recommendations immediately without asking further questions, using the health problem from the conversation context or user input
- If the user requests remedies or products (e.g., contains "remedies," "products," "give me"), provide recommendations immediately, prioritizing products from the dataset provided.
- For health problems, ask up to TWO relevant clarifying questions to understand the condition better, unless:
  - The user explicitly requests remedies/products.
  - You have enough information from prior responses (e.g., after two questions or clear context).
  - Stop asking questions after two have been asked and proceed to recommendations.
- Condition-specific question guidance:
  - For "hair fall" or "hairfall": Ask about symptoms (e.g., dandruff, itchiness) or lifestyle factors (e.g., stress, diet).
  - For "PCOS": Ask about symptoms (e.g., irregular periods, facial hair) or diet/lifestyle.
  - For "acne": Ask about symptoms (e.g., acne location, skin type) or triggers (e.g., diet, skincare).
  - For other health problems, ask symptom-based or lifestyle-related questions relevant to the condition. Ask only TWO specific questions.
- Only recommend products from the dataset when possible. If no dataset products match, say so and suggest checking back later.
- When recommending, note products are not currently available on Momy's Farm but can be found on Amazon, Flipkart, Patanjali, Nykaa, or 1mg.
- End recommendations with "Would you like help with anything else?"
- If the user says "reset", clear the conversation history.
- Do not provide specific URLs for external sites.

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
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return "Oops! Couldn't understand your request. Can you please rephrase it?"
    except Exception as e:
        print("Groq API Error:", e)
        return "I’m having trouble right now. Please try again later!"

def normalize_text(text):
    return re.sub(r"\s+", "", text.lower().strip())

def preprocess_data(df):
    df['normalized_problem'] = df['Health Problem'].apply(normalize_text)
    df['embedding'] = df['normalized_problem'].apply(lambda x: model.encode(x))
    return df

def extract_keyphrases(text, num_phrases=3):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=num_phrases)
    phrases = [kw[0] for kw in keywords]
    # Prioritize health-related phrases over generic ones like "rice"
    for phrase in phrases:
        if any(keyword in phrase.lower() for keyword in ["digestive", "heart", "diabetes", "health"]):
            return phrase
    return phrases[0] if phrases else text

def extract_products_from_response(response, df):
    dataset_products = set(df['Product'].str.lower())
    products = []
    lines = response.split('\n')
    for line in lines:
        numbered_match = re.match(r'^\d+\.\s*([^\:]+)\:', line.strip())
        if numbered_match:
            product = numbered_match.group(1).strip()
            if product.lower() in dataset_products:
                products.append(product)
        for product in dataset_products:
            if product in line.lower() and product not in [p.lower() for p in products]:
                products.append(product.title())
    bold_matches = re.findall(r'\*\*([^\*]+)\*\*', response)
    for match in bold_matches:
        if match.lower() in dataset_products and match not in products:
            products.append(match)
    return list(set(products))

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

    # Load and preprocess data from MongoDB
    df = load_products()
    if df.empty:
        return "Error: No products found in the database. Please check MongoDB."
    df = preprocess_data(df)

    # Handle reset request
    if user_input.lower().strip() == "reset":
        conversation = {"state": "initial", "health_problem": None, "context": [], "question_count": 0}
        session["conversation"] = conversation
        session.modified = True
        print("Conversation reset:", conversation)
        return "Let's start fresh! What's your health concern?"

    # Check for direct request for remedies/products
    direct_request = any(kw in user_input.lower() for kw in ["remedies", "products", "give me", "don’t ask", "just tell me"])

    # Build conversation context
    conversation_context = "\n".join([f"User: {c['user']}\nBot: {c['bot']}" for c in conversation["context"]])

    # Extract keyphrase and normalize
    keyphrase = extract_keyphrases(user_input)
    normalized_keyphrase = normalize_text(keyphrase)

    # Fuzzy matching for health problems (handle typos like "hearth")
    health_problems = df["normalized_problem"].unique()
    best_match = None
    best_score = 0
    for problem in health_problems:
        score = fuzz.ratio(normalized_keyphrase, problem)
        if score > 80 and score > best_score:  # Threshold for fuzzy match
            best_match = problem
            best_score = score

    # Detect health problem if initial
    if conversation["state"] == "initial":
        if best_match:
            conversation["state"] = "questioning"
            conversation["health_problem"] = best_match
            conversation["question_count"] = 1
        else:
            for problem in df["Health Problem"].str.lower():
                if problem in user_input.lower():
                    conversation["state"] = "questioning"
                    conversation["health_problem"] = normalize_text(problem)
                    conversation["question_count"] = 1
                    break

    # Update question count and state
    if conversation["state"] == "questioning" and not direct_request:
        conversation["question_count"] += 1
        if conversation["question_count"] >= 2 or any(kw in user_input.lower() for kw in ["recommend", "suggest", "try", "use"]):
            conversation["state"] = "recommendation"
    elif direct_request:
        conversation["state"] = "recommendation"
        conversation["question_count"] = 0

    # Prepare dataset products for LLM
    dataset_products = "\n".join([f"- {row['Product']}: {row['Health Benefit']} (for {row['Health Problem']})" for _, row in df.iterrows()])

    # Get LLM response with dataset context
    llm_response = get_groq_reply(user_input, profile, conversation_context, dataset_products)
    conversation["context"].append({"user": user_input, "bot": llm_response})

    session["conversation"] = conversation
    session.modified = True
    print("Updated conversation:", conversation)

    # Handle recommendation mode
    response = llm_response
    if conversation["state"] == "recommendation":
        # Step 1: Exact string matching using conversation health_problem or fuzzy match
        top_matches = pd.DataFrame()
        if conversation["health_problem"]:
            top_matches = df[df["normalized_problem"] == conversation["health_problem"]]
        if top_matches.empty and best_match:
            top_matches = df[df["normalized_problem"] == best_match]

        # Step 2: Fallback to embedding-based matching
        if top_matches.empty:
            user_embedding = model.encode(normalized_keyphrase)
            df["similarity"] = df["embedding"].apply(lambda x: util.cos_sim(x, user_embedding).item())
            top_matches = df[df["similarity"] > 0.5].sort_values(by="similarity", ascending=False)

        # Step 3: Filter for rice if requested
        is_rice_request = "rice" in user_input.lower()
        if is_rice_request and not top_matches.empty:
            top_matches = top_matches[top_matches["Product"].str.lower().str.contains("rice", na=False)]

        shown = set()
        if not top_matches.empty:
            response = "<br><br><strong>Here are some natural product recommendations for your concern:</strong><br><br>"
            for _, row in top_matches.iterrows():
                product = row['Product']
                benefit = row['Health Benefit']
                unique_key = f"{product.lower().strip()}|{benefit.lower().strip()}"
                if unique_key not in shown:
                    response += f"""<div style="margin-bottom: 12px;">
                        <strong>{product}</strong> – {benefit}<br>
                    </div>"""
                    shown.add(unique_key)
            response += "<br>These products are not currently available on Momy's Farm. We'll notify you once they are in stock!<br>"
            response += "In the meantime, you can find these products on sites like Amazon, Flipkart, Patanjali, Nykaa, or 1mg."
        else:
            response = llm_response + "<br><br>Note: I couldn’t find specific rice recommendations in our database for this concern. Would you like me to search again or suggest general remedies?"

        conversation["state"] = "completed"
        session["conversation"] = conversation
        session.modified = True

    # Handle purchase requests
    if "link" in user_input.lower() or "buy" in user_input.lower() or "purchase" in user_input.lower():
        products = extract_products_from_response(llm_response, df)
        if not products and top_matches.empty:
            response += "<br><br>I couldn’t identify specific products to recommend. Please clarify the products or health concern."
        else:
            response += "<br><br><strong>Where to Find Products:</strong><br><br>"
            response += "These products are not currently available on Momy's Farm. We'll notify you once they are in stock!<br>"
            response += "In the meantime, you can find these products on sites like Amazon, Flipkart, Patanjali, Nykaa, or 1mg."

    print("Response sent:", response)
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