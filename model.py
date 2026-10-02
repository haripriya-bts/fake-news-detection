import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


data = {
    "text": [
        "The government announced a new education policy for schools",
        "Scientists published a study about climate change",
        "The central bank announced a new interest rate decision",
        "The health department released a vaccination update",
        "The university announced the results of its annual examination",
        "The city council approved a new public transport project",
        "NASA published information about a new space mission",
        "The weather department issued a storm warning",
        "Scientists discovered a new planet in a distant solar system",
        "The election commission published official voting information",

        "Aliens landed in New York yesterday and took over the city",
        "Celebrity secretly turned into a vampire at midnight",
        "Scientists confirmed that humans can live without sleep forever",
        "A magic fruit gives people the ability to fly",
        "The government confirmed that the moon is made of cheese",
        "Drinking only chocolate can cure every disease",
        "A secret machine can make unlimited money instantly",
        "A famous actor said the sun will disappear tomorrow",
        "Cats were officially declared the rulers of the world",
        "Scientists said eating one coin makes people immortal"
    ],

    "label": [
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    lowercase=True
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)


def predict_news(text):
    features = vectorizer.transform([text])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    confidence = round(float(max(probabilities)) * 100, 2)

    if prediction == 1:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    return result, confidence