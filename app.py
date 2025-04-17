from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
from keybert import KeyBERT
import re

app = Flask(__name__, template_folder='templates')

# Load data
df = pd.read_csv("data/product_dataset.csv")

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg")

    if not user_input or user_input.strip() == "":
        return "Please describe your health concern so I can help you."

    # ✅ Extract and normalize user query
    keyphrase = extract_keyphrases(user_input)
    normalized_keyphrase = normalize_text(keyphrase)
    print(f"[Extracted Keyphrase]: {keyphrase} | [Normalized]: {normalized_keyphrase}")

    # ✅ Create embedding and compare
    user_embedding = model.encode(normalized_keyphrase)
    df["similarity"] = df["embedding"].apply(lambda x: util.cos_sim(x, user_embedding).item())
    top_matches = df[df["similarity"] > 0.6].sort_values(by="similarity", ascending=False)

    response = ""

    if not top_matches.empty:
        response += "Sure, I can help you with that! Here are some natural product recommendations:<br><br>"
        shown = set()

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
        # ✅ Trusted source fallback
        search_term = keyphrase.replace(" ", "+")
        response += "I couldn't find any exact matches in our product list, but here are some trusted sources you can explore:<br><br>"
        response += (
            f'🔗 <a href="https://www.amazon.in/s?k=natural+products+for+{search_term}" target="_blank">Amazon</a><br>'
            f'🔗 <a href="https://www.flipkart.com/search?q=natural+products+for+{search_term}" target="_blank">Flipkart</a><br>'
            f'🔗 <a href="https://www.patanjaliayurved.net/search?query={search_term}" target="_blank">Patanjali Ayurved</a><br>'
        )

    return response

if __name__ == "__main__":
    app.run(debug=True, port=5000)
