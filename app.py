from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup

app = Flask(__name__, template_folder='templates')

# Load data
df = pd.read_csv("data/product_dataset.csv")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all rows
df['embedding'] = df['Health Problem'].apply(lambda x: model.encode(x))

# Loosened filter in this function
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
            # Only skip very short or irrelevant-looking text
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

    # Keyword matching from Tags
    user_input_lower = user_input.lower()
    for _, row in df.iterrows():
        if 'Tags' in df.columns and any(word in user_input_lower for word in row['Tags'].lower().split(',')):
            return f"I recommend: <strong>{row['Product']}</strong> – {row['Description']}"

    # Semantic similarity logic
    user_embedding = model.encode(user_input)
    df["similarity"] = df["embedding"].apply(lambda x: util.cos_sim(x, user_embedding).item())
    top_matches = df[df["similarity"] > 0.4].sort_values(by="similarity", ascending=False)

    if not top_matches.empty:
        response = "Here are some natural product recommendations:<br><br>"
        seen = set()
        count = 0
        for _, row in top_matches.iterrows():
            key = f"{row['Product']} – {row['Health Benefit']}"
            if key not in seen:
                seen.add(key)
                response += f"<strong>{row['Product']}</strong> – {row['Health Benefit']}<br>"
                count += 1
            if count == 3:
                break
        return response
    else:
        return fetch_external_recommendations(user_input)

if __name__ == "__main__":
    app.run(debug=True, port=5000)