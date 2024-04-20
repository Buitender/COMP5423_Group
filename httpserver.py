from flask import Flask, request
from flask_cors import CORS

from chatbot import BartChatbot, ApiChatbot

app = Flask(__name__)
CORS(app, supports_credentials=True)

bart_chatbot = BartChatbot()
api_chatbot = ApiChatbot()


@app.route('/send2bart')
def receive_bart():
    msg = request.args.get('message')
    response = bart_chatbot.chat(msg)
    return response


@app.route('/send2api')
def receive_api():
    msg = request.args.get('message')
    response = api_chatbot.call_with_messages(input_text=msg)
    return response


if __name__ == '__main__':
    app.run()
