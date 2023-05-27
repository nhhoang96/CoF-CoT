import random
import os
import faiss
import uuid
import argparse
import numpy as np
import pandas as pd
from utils import io_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

vectorizer = TfidfVectorizer()
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def random_demo(demos, seed = 1, num=6):
    random.seed(seed)
    select_demos = random.choices(demos, k=num)
    return select_demos

def reasoning_type_demo(demos, num=6):
    demos = pd.DataFrame(demos)
    type_sample_num = int(num / demos.groupby('reasoning_type').ngroups)
    type_random_demos = demos.groupby('reasoning_type').apply(lambda x: x.sample(type_sample_num))
    type_random_demos.reset_index(drop=True, inplace=True)
    return type_random_demos.to_dict(orient='records')

def get_sparse_embedding(demos):
    question_embeddings = vectorizer.fit_transform([demo['question'] for demo in demos])
    return question_embeddings.toarray().astype(np.float32)

def get_dense_embedding(demos):
    demo_que_list = []
    for demo in demos:
        demo_que_list.append(demo['question'])
    question_embeddings = model.encode(demo_que_list, batch_size=20, show_progress_bar = True)
    return question_embeddings

def build_save_index(demo_que_embeddings, folder_path):
    index = faiss.IndexFlatL2(demo_que_embeddings.shape[1])
    index.add(np.array(demo_que_embeddings, dtype=np.float32))
    faiss.write_index(index, folder_path)

def load_index(folder_path: str):
    index = faiss.read_index(folder_path)
    return index

def retrieve_semantic_demo(demos, index, query_embedding, num = 6):
    select_demos = []
    scores, indices = index.search(np.array(query_embedding, dtype=np.float32), num)
    for j, i in enumerate(indices[0]):
        if i == -1:
            # This happens when not enough docs are returned.
            continue
        select_demos.append(demos[i])
    return select_demos

def centroid_demo(demos, all_que_embeding, index, num = 6):
    avg_question = np.mean(all_que_embeding, axis=0)
    scores, indices = index.search(np.array([avg_question], dtype=np.float32), num)
    select_demos = []
    for j, i in enumerate(indices[0]):
        if i == -1:
            # This happens when not enough docs are returned.
            continue
        select_demos.append(demos[i])
    return select_demos

def complex_cot_demo(demos):

    return demos

def build_demo_prompt_input(select_demos):
    prompt = ""
    for demo in select_demos:
        prompt += "question: " + demo['question'] + " logical form: " + demo['logical_form'] + "\n"
    return prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique", choices=["musique", "hotpotqa", "wiki2hopqa"], type=str,
                        help='Type of dataset')
    parser.add_argument("--add_demo", choices=['true', 'false'], default='false', type=str)
    parser.add_argument("--number_demos", default=6, type=int)
    parser.add_argument("--number_self_consistent_output", default=5, type=int)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--demo_select_strategy", default='random', type=str,
                        help='Kind of conditioning: (none, condition)')
    args = parser.parse_args()

    index_folder = "../data/Faiss_Index/musique/"
    train_data = io_utils.load_json(f"/export/home/musique/data/Unsupervised_QD/musique/train_withlf.json")
    dev_data = io_utils.load_json(f"/export/home/musique/data/Unsupervised_QD/musique/dev_withlf.json")
    demo_data = [
        {'question': line['question'], 'logical_form': line['logical_form'],
         'reasoning_type': line['id'].split('__')[0]} for
        line in train_data]

    if args.demo_select_strategy == "random":
        random_select_demo = random_demo(demo_data, num=args.number_demos)
        select_demo_prompt = build_demo_prompt_input(random_select_demo)

    elif args.demo_select_strategy == "random_within_same_type":
        random_within_type = reasoning_type_demo(demo_data, num=args.number_demos)
        select_demo_prompt = build_demo_prompt_input(random_within_type)

    elif args.demo_select_strategy == "sparse_retrieve":
        if not os.path.exists(index_folder + "sparse_index.faiss"):
            sparse_demo_que_embed = get_sparse_embedding(demo_data)
            build_save_index(sparse_demo_que_embed, index_folder + "sparse_index.faiss")

    elif args.demo_select_strategy == "dense_retrieve":
        if not os.path.exists(index_folder + "dense_index.faiss"):
            dense_demo_que_embed = get_dense_embedding(demo_data)
            build_save_index(dense_demo_que_embed, index_folder + "dense_index.faiss")

    elif args.demo_select_strategy == "sparse_centroid_demo":
        if not os.path.exists(index_folder + "sparse_index.faiss"):
            sparse_demo_que_embed = get_sparse_embedding(demo_data)
            build_save_index(sparse_demo_que_embed, index_folder + "sparse_index.faiss")
        all_test_que_embedding = get_sparse_embedding(dev_data)
        sparse_index = load_index(index_folder + "sparse_index.faiss")
        sparse_centroid_demo = centroid_demo(demo_data, all_test_que_embedding, sparse_index, num=args.number_demos)
        select_demo_prompt = build_demo_prompt_input(sparse_centroid_demo)

    elif args.demo_select_strategy == "dense_centroid_demo":
        if not os.path.exists(index_folder + "dense_index.faiss"):
            dense_demo_que_embed = get_dense_embedding(demo_data)
            build_save_index(dense_demo_que_embed, index_folder + "dense_index.faiss")
        all_test_que_embedding = get_dense_embedding(dev_data)
        dense_index = load_index(index_folder + "dense_index.faiss")
        dense_centroid_demo = centroid_demo(demo_data, all_test_que_embedding, dense_index, num=args.number_demos)
        select_demo_prompt = build_demo_prompt_input(dense_centroid_demo)

    elif args.demo_select_strategy == "complex_cot_demo":
        complex_demo = complex_cot_demo(demo_data)
        select_demo_prompt = build_demo_prompt_input(complex_demo)