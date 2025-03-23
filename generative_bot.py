from collections import deque
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utilities import generate_response
import pandas as pd
import numpy as np
import pickle
from utilities import encode_rag, cosine_sim_rag, top_candidates


class ChatBot:
    def __init__(self):
        self.conversation_history = deque([], maxlen=10)
        self.generative_model = None
        self.generative_tokenizer = None
        self.vect_data = []
        self.scripts = []
        self.ranking_model = None

    def load(self):
        """ "This method is called first to load all datasets and
        model used by the chat bot; all the data to be saved in
        tha data folder, models to be loaded from hugging face"""

        with open("data/spock_lines_vectorized.pkl", "rb") as fp:
            self.vect_data = pickle.load(fp)
        self.scripts = pd.read_pickle("data/spock_lines.pkl")
        self.ranking_model = SentenceTransformer(
            "greatakela/gnlp_hw1_encoder"
        )
        self.generative_model = AutoModelForSeq2SeqLM.from_pretrained(
            "greatakela/flan-t5-small-gen-chat_v3"
        )
        self.generative_tokenizer = AutoTokenizer.from_pretrained(
            "greatakela/flan-t5-small-gen-chat_v3"
        )

    def generate_answer(self, utterance):

        query_encoding = encode_rag(
            texts=utterance,
            model=self.ranking_model,
            contexts=self.conversation_history,
        )

        print("Query Encoding Shape:", query_encoding.shape)
        print("Stored Embeddings Shape:", np.array(self.vect_data).shape)


        bot_cosine_scores = cosine_sim_rag(
            self.vect_data,
            query_encoding,
        )

        top_scores, top_indexes = top_candidates(
            bot_cosine_scores, initial_data=self.scripts
        )

        print(top_scores, top_indexes)  # for debugging


        if top_scores[0] >= 0.9:
            for index in top_indexes:
                rag_answer = self.scripts.iloc[index]["ANSWER"]

            answer = generate_response(
                model=self.generative_model,
                tokenizer=self.generative_tokenizer,
                question=utterance,
                context=self.conversation_history,
                top_p=0.9,
                temperature=0.95,
                rag_answer=rag_answer,
            )
        else:
            answer = generate_response(
                model=self.generative_model,
                tokenizer=self.generative_tokenizer,
                question=utterance,
                context=self.conversation_history,
                top_p=0.9,
                temperature=0.95,
            )

        self.conversation_history.append(utterance)
        self.conversation_history.append(answer)
        return answer