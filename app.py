from flask import Flask, render_template, request
from model import predict_news

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    message = None

    if request.method == "POST":
        news = request.form.get("news", "").strip()

        if not news:
            message = "Please enter a news headline or article."
        else:
            result, confidence = predict_news(news)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        message=message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)