from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

app = Flask(__name__)

# Load data
data = pd.read_csv("products.csv")

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['description'])

search_history = []

def recommend_products(keyword):
    keyword_vec = tfidf.transform([keyword])
    similarity = cosine_similarity(keyword_vec, tfidf_matrix)
    data['score'] = similarity[0]
    return data.sort_values(by='score', ascending=False).head(10)

def correct_typo(keyword):
    choices = data['name'].tolist()
    best_match = process.extractOne(keyword, choices)
    return best_match[0] if best_match else keyword

@app.route("/")
def index():
    recommendations = []
    if search_history:
        keyword = max(set(search_history), key=search_history.count)
        recommendations = recommend_products(keyword).to_dict(orient='records')
    return render_template("index.html", recommendations=recommendations)

@app.route("/search", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        keyword = request.form['keyword']
        corrected = correct_typo(keyword)
        search_history.append(corrected)

        keyword_vec = tfidf.transform([corrected])
        similarity = cosine_similarity(keyword_vec, tfidf_matrix)

        data['score'] = similarity[0]

        results = (
            data[data['score'] > 0.05]
            .sort_values(by='score', ascending=False)
            .head(15)
            .to_dict(orient='records')
        )

    return render_template("search.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
