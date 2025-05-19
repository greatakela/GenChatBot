<p align="center">
  <img src="https://www.merchandisingplaza.co.uk/282130/2/Stickers-Star-Trek-STAR-TREK-Spock-Live-Long-Prosper-Sticker-l.jpg" 
       alt="Live Long and Prosper Sticker" 
       width="300">
</p>

# HW2 Generative Model-Based Chat Bot

**Task**: You need to develop a chatbot using a generative approach. The bot should carry on a conversation as a specific character from a series, imitating that character's speech style and mannerisms. It's important to account for the character’s speech patterns, the topics they discuss, and their typical reactions.

## Data Collection
As a foundation for the chatbot, I used scripts from the *Star Trek* series, which I downloaded from this [repository](https://github.com/varenc/star_trek_transcript_search), particularly the lines of Mr. Spock, a crew member and scientist from the planet Vulcan.

### Data for the Retrieval Component of the Chatbot

Initial data processing was similar to what I did for the previous homework assignment (retrieval-based chatbot):

- Script cleaning  
- Selecting the character’s lines as bot response candidates  
- Extracting the previous line as the question (empty if first in the scene)  
- Extracting earlier lines as dialogue context (empty if first in the scene)

To improve factual consistency, context-aware embeddings were used. To preserve thematic and stylistic coherence, generation is augmented with retrieval results based on cosine similarity between the user’s context-question and stored data (as in the previous HW). All base data is vectorized into a database (file `spock_lines_vectorized.pkl`). For vectorization, I used the bi-encoder model trained in HW1, hosted on Hugging Face ([link](https://huggingface.co/greatakela/gnlp_hw1_encoder)).

### Data for the Generative Model

For training the generative model, I reused the same preprocessed data as for retrieval, but applied augmentation by splitting the context into parts. For a context with 3 sentences, 4 samples are created:

- answer + question + sentence 3 + sentence 2 + sentence 1  
- answer + question + sentence 3 + sentence 2  
- answer + question + sentence 3  
- answer + question  

This resulted in about 38,000 training samples, saved in `spock_lines_context.pkl`.

Data prep code: [GNLP_HW2_data_prep.ipynb](https://github.com/greatakela/GenChatBot/blob/main/Notebooks/GNLP_HW2_data_prep.ipynb)

## Chatbot Architecture

Workflow of the chatbot is illustrated below.

![image](https://github.com/greatakela/GenChatBot/blob/main/static/ArchGenBot.png)

### Retrieval Component

**Reply database** — vectorized scripts using a [trained encoder](https://huggingface.co/greatakela/gnlp_hw1_encoder), including context and question. Model details are in HW1. Here, the pre-trained model is reused ([link](https://huggingface.co/greatakela/gnlp_hw1_encoder)).

Top-1 matching reply (based on cosine similarity) is passed to the generative model as context, using the **RAG (retrieval-augmented generation)** strategy.

Threshold for inclusion is **0.9**. If cosine similarity is lower, the reply is not passed to the model.

### Generative Component

The main part of the chatbot is the generative model. Given the dataset size, I fine-tuned a small T5-family model (`google/flan-t5-base`, 248M parameters) [model card](https://huggingface.co/google/flan-t5-base). Training was done over 5 epochs on Colab with A100. Notebook: [training](https://github.com/greatakela/GenChatBot/blob/main/Notebooks/GNLP_HW2_FLAN_T5_train_model.ipynb)

Input:  
```text
"context: " + context + "</s>question: " + question  
```
### Training Evaluation

During training, standard metrics such as train and evaluation loss were logged. In addition, I implemented automatic metrics to measure the similarity between the generated responses and the target responses from the original script. 

While automatic metrics are often criticized and generally not sufficient for evaluating generative models on their own, I used them as directional indicators to assess whether continued training was necessary. These metrics should, of course, be complemented with human-based evaluations—especially when fine-tuning generation strategies.

For automatic evaluation, I used the `evaluate` library from Hugging Face, specifically the [**ROUGE**](https://huggingface.co/spaces/evaluate-metric/rouge) and [**BERTScore**](https://huggingface.co/spaces/evaluate-metric/bertscore) metric packages:

- **ROUGE-1** – unigram overlap between the generated text and the target (higher means more similar)
- **ROUGE-2** – bigram overlap (higher means more similar)
- **ROUGE-L** – longest common subsequence match (higher means better structural similarity)
- **ROUGE average generated length** – average length of the generated responses (useful to understand how verbose the model is)
- **BERTScore Recall** – cosine similarity between target and generated embeddings (closer to 1 indicates stronger recall)
- **BERTScore Precision** – cosine similarity between generated and target embeddings (closer to 1 indicates stronger precision)
- **BERTScore F1** – harmonic mean of BERTScore precision and recall (closer to 1 indicates higher overall similarity)

Below are screenshots from Weights & Biases showing how these metrics evolved during training:

**Graphs:**

ROUGE:

<img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_rouge_1.png" width="32.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_rouge_2.png" width="32.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_rouge_l.png" width="32.5%">

BERTScore:

<img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_bs_r.png" width="32.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_bs_p.png" width="32.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_bs_f1.png" width="32.5%">

Loss:

<img src="https://github.com/greatakela/GenChatBot/blob/main/static/train_loss.png" width="49.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_loss.png" width="49.5%">

### Model Training Summary

The results indicate that the model performed very well during both training and validation.  
Loss values for both the training and validation datasets decreased steadily with each epoch, suggesting that the model was effectively learning and adapting to the task.

The consistent reduction in loss reflects improved predictive performance. A significant drop in training loss indicates successful learning and pattern recognition from the training data, while the decreasing validation loss confirms that the model is generalizing well without overfitting.

The narrowing gap between training and validation loss over time is also a positive sign. That said, it’s important to monitor this difference carefully—if the gap becomes too small, it could indicate underfitting; if too large, potential overfitting.

In conclusion, the decreasing and converging training and validation losses are strong indicators of the model's ability to generalize and respond effectively to unseen data, which is critical for text generation tasks.

Training graphs show there's still room for fine-tuning, as both eval and train losses continued to decline.  
Despite this potential, training was stopped after 5 epochs, as text similarity metrics began to stabilize—though they continued to show gradual improvement.

### Generation Strategy Tuning

To determine optimal generation parameters for the chatbot, I ran several experiments while adjusting key generation settings.  
You can view the experiment notebook [here](https://github.com/greatakela/GenChatBot/blob/main/Notebooks/GNLP_HW2_generation_evaluation.ipynb).

After testing, I chose the following as fixed parameters:
- `do_sample=True` – adds randomness to generation
- `max_length=1000` – no hard limit on output length
- `repetition_penalty=2.0` – mitigates repetition due to slight undertraining
- `top_k=50` – values lower than this reduce the model’s responsiveness to user input
- `no_repeat_ngram_size=2` – further helps control repetition

I experimented with `top_p` and `temperature` to evaluate their effect on creativity and text variation.  
Evaluation was based on cosine similarity between generated responses and target script replies, using a random sample of 100 items from `spock_lines_context.pkl`.  
Similarity was measured using the same bi-encoder model used in the retrieval component of the chatbot.  
I also tracked response generation time.

The tested parameter combinations were:

- **temperature = 0.2, top_p = 0.1** – expected safe, generic outputs, possibly lacking character personality  
- **temperature = 0.5, top_p = 0.5** – standard responses with slightly more expressive variability  
- **temperature = 0.7, top_p = 0.8** – more creativity, with emerging character traits  
- **temperature = 0.9, top_p = 0.9** – stronger creativity, clear expression of character style  
- **temperature = 1.0, top_p = 0.95** – highest creativity, but with increased risk of drifting off-topic


Cosine similarity and generation time plots:

<img src="https://github.com/greatakela/GenChatBot/blob/main/static/gen_time.png" width="49.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/cos_sim.png" width="49.5%">

**Best results**: `temperature=0.9`, `top_p=0.9` for style diversity with acceptable factual consistency

## Repository Structure

```bash
│   README.md - HW2 report
│   requirements.txt
│   .gitignore
│   __init__.py
│   generative_bot.py - main logic
│   utilities.py - helper functions
│   app.py - Flask UI launcher
│
├───Notebooks - training & evaluation notebooks
├───templates - web interface layout
│       chat.html
├───static - web interface styles
│       style.css
├───data
│       spock_lines_context.pkl - processed context-augmented training data
│       spock_lines_vectorized.pkl - vectorized context-question DB
│       spock_lines.pkl - raw data
```
## Web Service

The chatbot uses Flask, launched via `app.py`, which sets up the UI, loads models, and handles requests.

To run locally:
```bash
git clone https://github.com/greatakela/GenChatBot.git
python -m venv venv
pip install -r requirements.txt
python app.py
```
Accessible at: `http://127.0.0.1:5000`

### Asynchronous Handling in the Flask Application

Flask supports asynchronous behavior by allowing the use of asynchronous route handlers, enabling event-driven concurrency through `async` and `await`.  
When a request hits an asynchronous route, Flask runs its processing loop with each event handled in a separate thread or coroutine.

In my implementation, the Flask app handles only two types of events:
- Rendering the interface
- Receiving a user request and generating a response (this part cannot be asynchronous, as a response requires the user's input first)

To demonstrate asynchronous capabilities in the app’s codebase, I added a small auxiliary coroutine that runs in parallel with response generation—a simple sleep operation:


```python
async def sleep():
    await asyncio.sleep(0.1)
    return 0.1

@app.route("/get", methods=["GET", "POST"])
async def chat():
    msg = request.form["msg"]
    input = msg
    await asyncio.gather(sleep(), sleep())
    return get_Chat_response(input)
```
### Gunicorn & Gevent for Multiprocessing

Gunicorn command for async multi-worker deployment:

```bash
gunicorn --timeout 1000 --workers 2 --worker-class gevent --worker-connections 100 app:app -b 0.0.0.0:5000
```
This launches Gunicorn with 2 workers and 50 async connections per worker.

## Conclusion

The generative model demonstrated high effectiveness, with clear generalization potential. To better evaluate its full capabilities, further experiments with more diverse data are needed.

## Online Deployment

The Dockerized project was deployed on a **Kamatera** virtual server. The chatbot is available at:

👉 http://185.53.209.56:5000/

Docker image was optimized to < 2GB.  
VM specs: 2 CPU, 2 GB RAM, 80 GB disk
