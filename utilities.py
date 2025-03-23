import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pickle


# ===================================================
def encode(texts, model, contexts=None, do_norm=True):
    """function to encode texts for cosine similarity search"""

    question_vectors = model.encode(texts)
    if type(contexts) is list:
        context_vectors = model.encode("".join(contexts))
    else:
        context_vectors = model.encode(contexts)

    return np.concatenate(
        [
            np.asarray(context_vectors),
            np.asarray(question_vectors),
        ],
        axis=-1,
    )


def encode_rag(texts, model, contexts=None, do_norm=True):
    """function to encode texts for cosine similarity search"""

    question_vectors = model.encode(texts)

    # if contexts is None:
    #     context_vectors = model.encode("")  # Encode an empty string instead
    # else:
    #     context_vectors = model.encode("".join(contexts))

    context_vectors = model.encode("".join(contexts))

    return np.concatenate(
        [
            np.asarray(context_vectors),
            np.asarray(question_vectors),
        ],
        axis=-1,
    )


# ===================================================
def cosine_sim(answer_true_vectros, answer_generated_vectors) -> list:
    """FOR MODEL EVALUATION!!!!
    returns list of tuples with similarity score"""

    data_emb = sparse.csr_matrix(answer_true_vectros)
    query_emb = sparse.csr_matrix(answer_generated_vectors)
    similarity = cosine_similarity(query_emb, data_emb).flatten()
    return similarity[0]


# ===================================================
def cosine_sim_rag(data_vectors, query_vectors) -> list:
    """FOR RAG RETRIEVAL RANKS!!!
    returns list of tuples with similarity score and
    script index in initial dataframe"""

    data_emb = sparse.csr_matrix(data_vectors)
    query_emb = sparse.csr_matrix(query_vectors)
    similarity = cosine_similarity(query_emb, data_emb).flatten()
    ind = np.argwhere(similarity)
    match = sorted(zip(similarity, ind.tolist()), reverse=True)

    return match


# ===================================================
def generate_response(
    model,
    tokenizer,
    question,
    context,
    top_p,
    temperature,
    rag_answer="",
):

    combined = (
        "context:" + rag_answer +
        "".join(context) + "</s>" +
        "question: " + question
    )
    input_ids = tokenizer.encode(combined, return_tensors="pt")
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=1000,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=2.0,
        top_k=50,
        no_repeat_ngram_size=4,
        # early_stopping=True,
        # min_length=10,
    )

    out = tokenizer.decode(sample_output[0][1:], skip_special_tokens=True)
    if "</s>" in out:
        out = out[: out.find("</s>")].strip()

    return out


# ===================================================
def top_candidates(score_lst_sorted, initial_data, top=1):
    """this functions receives results of the cousine similarity ranking and
    returns top items' scores and their indices"""

    scores = [item[0] for item in score_lst_sorted]
    candidates_indexes = [item[1][0] for item in score_lst_sorted]
    return scores[0:top], candidates_indexes[0:top]