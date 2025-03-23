from flask import Flask, render_template, request
#from flask_ngrok import run_with_ngrok
from generative_bot import ChatBot
import asyncio

app = Flask(__name__)   
generative_chat = ChatBot()
generative_chat.load()
#run_with_ngrok(app)  # Expose the app via ngrok

# this script is running flask application


@app.route("/")
async def index():
    return render_template("chat.html")


async def sleep():
    await asyncio.sleep(0.1)
    return 0.1


@app.route("/get", methods=["GET", "POST"])
async def chat():
    msg = request.form["msg"]
    input = msg
    await asyncio.gather(sleep(), sleep())
    return get_Chat_response(input)


def get_Chat_response(text):
    answer = generative_chat.generate_answer(text)
    return answer


if __name__ == "__main__":
    app.run()