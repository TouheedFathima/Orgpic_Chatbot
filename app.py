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
import uuid

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__, template_folder='templates')
app.secret_key = "orgpick_secret_123"

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["momy_db"]
product_collection = db["products"]
profile_collection = db["profiles"]

# Cached DataFrame
cached_df = None

# Load data from MongoDB into a DataFrame
def load_products():
    global cached_df
    if cached_df is None:
        products = list(product_collection.find({}, {'_id': 0}))
        cached_df = pd.DataFrame(products)
        if not cached_df.empty:
            cached_df = preprocess_data(cached_df)
    return cached_df

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

def preprocess_data(df):
    df['normalized_problem'] = df['Health Problem'].apply(normalize_text)
    df['embedding'] = df['normalized_problem'].apply(lambda x: model.encode(x))
    df['category'] = df['Product'].apply(categorize_product)
    return df

def categorize_product(product_name):
    product_name = product_name.lower()
    if any(kw in product_name for kw in ['oil', 'ghee', 'butter']):
        return 'oils'
    elif any(kw in product_name for kw in ['flour', 'atta']):
        return 'flours'
    elif any(kw in product_name for kw in ['rice', 'samba', 'brown']):
        return 'rice'
    elif any(kw in product_name for kw in ['seed', 'flax', 'pumpkin']):
        return 'seeds'
    elif any(kw in product_name for kw in ['noodle', 'sevai', 'pasta']):
        return 'noodles'
    elif any(kw in product_name for kw in ['dal', 'lentil', 'peas', 'beans']):
        return 'pulses'
    elif any(kw in product_name for kw in ['tea', 'coffee', 'malt']):
        return 'beverages'
    elif any(kw in product_name for kw in ['jam', 'marmalade', 'sauce', 'ketchup']):
        return 'spreads'
    elif any(kw in product_name for kw in ['muesli', 'flakes', 'oats']):
        return 'cereals'
    elif any(kw in product_name for kw in ['almond', 'pistachio']):
        return 'nuts'
    elif any(kw in product_name for kw in ['masala', 'spice', 'powder']):
        return 'spices'
    elif any(kw in product_name for kw in ['shampoo', 'conditioner']):
        return 'shampoos'
    elif any(kw in product_name for kw in ['mask', 'hair mask']):
        return 'masks'
    elif any(kw in product_name for kw in ['snack', 'sweet ball']):
        return 'snacks'
    elif any(kw in product_name for kw in ['attar', 'perfume', 'scent']):
        return 'scents'
    elif any(kw in product_name for kw in ['face wash', 'facewash']):
        return 'facewash'
    else:
        return 'other'

def normalize_text(text):
    return re.sub(r"\s+", "", text.lower().strip())

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

Dataset Products (for reference in remedy/how-to-use responses):
{dataset_products or 'No specific products provided'}

Instructions:
- Use a warm, friendly, and concise tone. Greet the user by their first name ONLY in the first response of a new conversation.
- Keep responses short, readable, and well-structured:
  - Use short paragraphs (1-2 sentences).
  - For remedies, how-to-use, diet plans, or product explanations, provide a numbered list with clear, point-wise explanations (e.g., "1. Garlic: Add to cooking to lower cholesterol").
- If the user input is a greeting (e.g., "hi", "hello", "hlo"), respond kindly, reset the conversation, and ask how you can help.
- For health problems without an explicit product request, ask ONE relevant clarifying question to understand the concern unless you have enough information from prior responses.
- Condition-specific question guidance:
  - For "hair fall" or "dandruff": Ask about symptoms (e.g., itchiness, flakiness) or product preference (e.g., shampoos, oils).
  - For "PCOS": Ask about symptoms (e.g., irregular periods) or diet preferences.
  - For "acne": Ask about skin type or triggers (e.g., oily skin, diet).
  - For "digestive issues": Ask about symptoms (e.g., bloating, constipation).
- For diet plan requests (e.g., "diet plan," "meal plan"), provide a detailed, condition-specific diet plan in a numbered list, including foods to include and avoid, tailored to the user's health concern (e.g., PCOS-friendly foods).
- For non-product queries (e.g., "recipe," "benefit," "how to," "diet plan"), provide detailed general advice, recipes, or usage instructions in a numbered list without recommending products unless explicitly requested.
- For remedy or how-to-use queries (e.g., "remedies for hair fall," "how to use shampoo"), provide general remedies or usage instructions in a numbered list, referencing dataset products only if listed in the provided dataset_products.
- For product requests (e.g., "products," "give me," "recommend"), provide general advice or context about the health concern or product category without recommending specific products.
- If the user clarifies they don’t want products (e.g., "not products"), focus on general advice, recipes, or diet plans without product recommendations.
- End responses with "Would you like help with anything else?" when appropriate.
- Do not provide specific URLs for external sites.

Current User Message: {user_input}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
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
        raise

def extract_keyphrases(text, df, num_phrases=3):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=num_phrases)
    phrases = [kw[0] for kw in keywords]
    health_problems = set(df["normalized_problem"].str.lower())
    for phrase in phrases:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase in health_problems:
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
        return "Please describe your health concern or let me know how I can assist you."

    # Get or set session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.modified = True

    # Load profile from MongoDB
    profile = profile_collection.find_one({'session_id': session['session_id']}, {'_id': 0})
    if not profile:
        profile = {}

    conversation = session.get("conversation", {
        "state": "initial",
        "health_problem": None,
        "context": [],
        "question_count": 0,
        "specific_product_type": None,
        "recommended_products": [],
        "no_products": False
    })

    # Load and preprocess data
    df = load_products()
    if df.empty:
        return "Error: No products found in the database. Please check MongoDB."

    # Handle reset or start over
    if user_input.lower().strip() in ["reset", "start over"]:
        conversation = {"state": "initial", "health_problem": None, "context": [], "question_count": 0, "specific_product_type": None, "recommended_products": [], "no_products": False}
        session["conversation"] = conversation
        session.modified = True
        print("Conversation reset:", conversation)
        return "Let's start fresh! What's your health concern or how can I help?"

    # Handle greetings
    if user_input.lower().strip() in ["hi", "hello", "hlo", "hey"]:
        conversation = {"state": "initial", "health_problem": None, "context": [], "question_count": 0, "specific_product_type": None, "recommended_products": [], "no_products": False}
        session["conversation"] = conversation
        session.modified = True
        print("Greeting detected, conversation reset:", conversation)
        return f"Hey {profile.get('name', '')}! How can I assist you today?"

    # Check for query types
    non_health_query = any(kw in user_input.lower() for kw in ["recipe", "benefit", "how to", "what is", "diet plan", "meal plan"])
    product_request = any(kw in user_input.lower() for kw in ["products", "give me", "recommend", "suggest", "items", "list"]) and not any(kw in user_input.lower() for kw in ["diet", "plan"])
    remedy_query = any(kw in user_input.lower() for kw in ["remedies", "how to use", "consumption"])
    no_products_clarification = any(kw in user_input.lower() for kw in ["not products", "no products", "don’t want products"])

    # Detect specific product type
    specific_product_type = None
    for category in ["flours", "rice", "snacks", "shampoos", "oils", "seeds", "noodles", "pulses", "beverages", "spreads", "cereals", "nuts", "spices", "masks", "scents", "facewash"]:
        if category in user_input.lower() or category[:-1] in user_input.lower():
            specific_product_type = category
            break

    # Extract keyphrase and normalize
    keyphrase = extract_keyphrases(user_input, df)
    normalized_keyphrase = normalize_text(keyphrase)

    # Fuzzy matching for health problems
    health_problems = df["normalized_problem"].unique()
    best_match = None
    best_score = 0
    for problem in health_problems:
        score = fuzz.ratio(normalized_keyphrase, problem)
        if score > 70 and score > best_score:
            best_match = problem
            best_score = score

    # Handle conversation state
    if non_health_query or no_products_clarification:
        conversation["state"] = "initial"
        conversation["health_problem"] = None
        conversation["question_count"] = 0
        conversation["specific_product_type"] = specific_product_type
        conversation["recommended_products"] = []
        conversation["no_products"] = True
    elif product_request and best_match and (conversation["state"] in ["initial", "completed"] or conversation["health_problem"] != best_match):
        conversation["state"] = "recommendation"
        conversation["health_problem"] = best_match
        conversation["question_count"] = 0
        conversation["specific_product_type"] = specific_product_type
        conversation["recommended_products"] = []
        conversation["no_products"] = False
    elif remedy_query and conversation["recommended_products"]:
        conversation["state"] = "remedy"
        conversation["question_count"] = 0
        conversation["specific_product_type"] = specific_product_type
        conversation["no_products"] = False
    elif best_match and (conversation["state"] in ["initial", "completed"] or conversation["health_problem"] != best_match):
        conversation["state"] = "questioning"
        conversation["health_problem"] = best_match
        conversation["question_count"] = 0
        conversation["specific_product_type"] = specific_product_type
        conversation["recommended_products"] = []
        conversation["no_products"] = False
    elif not best_match and conversation["state"] in ["initial", "completed"]:
        for problem in df["Health Problem"].str.lower():
            if problem in user_input.lower():
                conversation["state"] = "questioning"
                conversation["health_problem"] = normalize_text(problem)
                conversation["question_count"] = 0
                conversation["specific_product_type"] = specific_product_type
                conversation["recommended_products"] = []
                conversation["no_products"] = False
                break
        else:
            if product_request:
                conversation["state"] = "recommendation"
                conversation["health_problem"] = normalized_keyphrase if normalized_keyphrase in health_problems else None
                conversation["question_count"] = 0
                conversation["specific_product_type"] = specific_product_type
                conversation["recommended_products"] = []
                conversation["no_products"] = False

    # Build conversation context (limit to last 5 exchanges)
    conversation_context = "\n".join([f"User: {c['user']}\nBot: {c['bot']}" for c in conversation["context"][-5:]])

    # Update state for questioning mode
    if conversation["state"] == "questioning":
        conversation["question_count"] += 1
        if conversation["question_count"] >= 1 or any(kw in user_input.lower() for kw in ["try", "use", "don’t ask", "just tell"]):
            conversation["state"] = "recommendation"

    # Prepare dataset products for remedy queries
    dataset_products = ""
    if conversation["state"] == "remedy" and conversation["recommended_products"]:
        relevant_products = df[df["Product"].isin(conversation["recommended_products"])]
        dataset_products = "\n".join([f"- {row['Product']}: {row['Health Benefit']} (for {row['Health Problem']}, category: {row['category']})" for _, row in relevant_products.iterrows()])

    # Check MongoDB for products first for product requests
    response = ""
    top_matches = pd.DataFrame()
    if conversation["state"] == "recommendation" and product_request and not conversation["no_products"]:
        if conversation["health_problem"]:
            top_matches = df[df["normalized_problem"] == conversation["health_problem"]]
        if top_matches.empty and best_match:
            top_matches = df[df["normalized_problem"] == best_match]
        if top_matches.empty:
            user_embedding = model.encode(normalized_keyphrase)
            df["similarity"] = df["embedding"].apply(lambda x: util.cos_sim(x, user_embedding).item())
            top_matches = df[df["similarity"] > 0.5].sort_values(by="similarity", ascending=False)

        if conversation["specific_product_type"]:
            if conversation["specific_product_type"] == "facewash":
                top_matches = top_matches[top_matches["Product"].str.lower().str.contains("face wash|facewash")]
            else:
                top_matches = top_matches[top_matches["category"] == conversation["specific_product_type"]]

        if not top_matches.empty:
            response = "<strong>Here are some natural product recommendations for your concern:</strong><br><br>"
            shown = set()
            conversation["recommended_products"] = []
            top_matches = top_matches.head(6)  # Limit to 6 products
            for _, row in top_matches.iterrows():
                product = row['Product']
                benefit = row['Health Benefit']
                unique_key = f"{product.lower().strip()}|{benefit.lower().strip()}"
                if unique_key not in shown:
                    response += f"""<div style="margin-bottom: 12px;">
                        <strong>{product}</strong> – {benefit}<br>
                    </div>"""
                    shown.add(unique_key)
                    conversation["recommended_products"].append(product)
            response += "<br>These products are not currently available on Momy's Farm. We'll notify you once they are in stock!<br>"
            response += "In the meantime, you can find these products on sites like Amazon, Flipkart, Patanjali, Nykaa, or 1mg."
            response += "<br>Would you like help with anything else?"
            conversation["state"] = "completed"
            session["conversation"] = conversation
            session.modified = True
            print("Response sent (MongoDB):", response)
            return response

    # Get LLM response for non-product queries or if no products found
    try:
        llm_response = get_groq_reply(user_input, profile, conversation_context, dataset_products if conversation["state"] == "remedy" else None)
    except Exception:
        if conversation["state"] == "recommendation" and product_request:
            response = "<br><br>Note: I couldn’t find specific products in our database for this concern, and I’m having trouble connecting to provide general advice. Please try again later!"
            conversation["state"] = "completed"
            session["conversation"] = conversation
            session.modified = True
            print("Response sent (fallback):", response)
            return response
        return "I’m having trouble right now. Please try again later!"

    # Store conversation
    conversation["context"].append({"user": user_input, "bot": llm_response})
    if len(conversation["context"]) > 10:
        conversation["context"] = conversation["context"][-10:]
    session["conversation"] = conversation
    session.modified = True
    print("Updated conversation:", conversation)

    # Initialize response with LLM response
    response = llm_response

    # Handle recommendation mode for product requests (if no MongoDB products)
    if conversation["state"] == "recommendation" and product_request and not conversation["no_products"]:
        conversation["state"] = "completed"
        session["conversation"] = conversation
        session.modified = True

    # Handle remedy queries
    if conversation["state"] == "remedy" and conversation["recommended_products"]:
        response = llm_response
        response += "<br>Would you like help with anything else?"
        conversation["state"] = "completed"
        session["conversation"] = conversation
        session.modified = True

    # Handle non-health or diet plan queries
    if non_health_query or conversation["no_products"]:
        response = llm_response
        response += "<br>Would you like help with anything else?"
        conversation["state"] = "completed"
        session["conversation"] = conversation
        session.modified = True

    # Handle purchase requests
    if any(kw in user_input.lower() for kw in ["link", "buy", "purchase"]):
        products = extract_products_from_response(llm_response, df) or conversation["recommended_products"]
        if not products:
            response += "<br><br>I couldn’t identify specific products to recommend. Please clarify the products or health concern."
        else:
            response += "<br><br><strong>Where to Find Products:</strong><br><br>"
            response += "These products are not currently available on Momy's Farm. We'll notify you once they are in stock!<br>"
            response += "In the meantime, you can find these products on sites like Amazon, Flipkart, Patanjali, Nykaa, or 1mg."

    print("Response sent:", response)
    return response

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.modified = True

    if request.method == "POST":
        profile_data = {
            'session_id': session['session_id'],
            'name': request.form.get("name"),
            'age': request.form.get("age"),
            'gender': request.form.get("gender"),
            'diet': request.form.get("diet"),
            'goal': request.form.get("goal"),
            'allergies': request.form.get("allergies"),
            'lifestyle': request.form.get("lifestyle"),
        }
        profile_collection.update_one(
            {'session_id': session['session_id']},
            {'$set': profile_data},
            upsert=True
        )
        return redirect("/")

    profile_data = profile_collection.find_one({'session_id': session.get('session_id')}, {'_id': 0})
    return render_template("profile.html", profile=profile_data or {})

@app.route("/")
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.modified = True

    profile = profile_collection.find_one({'session_id': session['session_id']}, {'_id': 0})
    if not profile:
        return redirect("/profile")
    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port, debug=True)
else:
    application = app