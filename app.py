from flask import Flask, render_template, request
from model.huggingface_utils import predict_fake_news

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'news' in request.form and request.form['news'].strip():
        # Handle text input (existing functionality)
        news = request.form['news']
        prediction, confidence, sources = predict_fake_news(news)

        result = "🟢 Real News" if prediction == 0 else "🔴 Fake News"
        confidence_percent = round(confidence * 100, 2)
        return render_template('result.html', news=news, result=result, confidence=confidence_percent, sources=sources)

    else:
        return render_template('result.html', error="No input provided")

if __name__ == '__main__':
    app.run(debug=True)
