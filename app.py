from flask import Flask, render_template, request, jsonify
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

# Create the question-answering pipeline
qna = pipeline('question-answering', model=model, tokenizer=tokenizer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    article = request.form['article']
    question = request.form['question']

    # Use the question-answering pipeline to answer the question
    result = qna(context=article, question=question)

    return render_template('index.html', answer=result['answer'])

if __name__ == '__main__':
    app.run(debug=True)
