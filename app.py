from flask import Flask, request, render_template
from toxic import preprocess, get_score


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html', level=' ....')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = str(get_score(text) * 100)
    return render_template('form.html', level=processed_text)
