# -*- coding: utf-8 -*-

import json
import os
import random
import re
import argparse
import openai
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
api_key = "******"
client = openai.OpenAI(api_key=api_key)

DATASET_MAP = {
    'cadec': {
        'example_files': ['/mnt/data0/LSDNER/data/cadec/train_full_labeled.jsonl'],
        'test_file': '/mnt/data0/LSDNER/data/cadec/test.jsonl',
        'label_semantic_file': '/mnt/data0/LSDNER/ICL/cadec/LSD/LSD',
        'diversity_example_files': [
            '/mnt/data0/LSDNER/ICL/cadec/cluster/cluster_1/sample1_labeled.jsonl',
            '/mnt/data0/LSDNER/ICL/cadec/cluster/cluster_2/sample2_labeled.jsonl'
        ],
        'output_dir': '/mnt/data0/LSDNER/ICL/cadec/output/',
        'label_set': ['ADR']
    }
}

def read_random_samples(file, num_samples_per_file):
    samples = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e} in line: {line}")
    if len(samples) > num_samples_per_file:
        return random.sample(samples, num_samples_per_file)
    else:
        return samples

def try_load_sentence_transformer(model_path_or_name):
    try:
        model = SentenceTransformer(model_path_or_name)
        return model
    except Exception as e:
        print(f'加载句嵌入模型失败: {e}')
        return None

def get_most_similar_samples(test_sentence, candidate_samples, num_samples, model):
    candidate_texts = [sample.get('sentence', sample.get('text', '')) for sample in candidate_samples]
    embeddings = model.encode(candidate_texts)
    test_embedding = model.encode([test_sentence])[0]
    similarities = np.dot(embeddings, test_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(test_embedding) + 1e-8
    )
    top_indices = np.argsort(similarities)[-num_samples:][::-1]
    return [candidate_samples[i] for i in top_indices]

def get_most_similar_samples_tfidf(test_sentence, candidate_samples, num_samples):
    candidate_texts = [sample.get('sentence', sample.get('text', '')) for sample in candidate_samples]
    texts = candidate_texts + [test_sentence]
    vectorizer = TfidfVectorizer().fit(texts)
    tfidf_matrix = vectorizer.transform(texts)
    test_vec = tfidf_matrix[-1]
    candidate_vecs = tfidf_matrix[:-1]
    similarities = cosine_similarity(candidate_vecs, test_vec)
    similarities = similarities.flatten()
    top_indices = np.argsort(similarities)[-num_samples:][::-1]
    return [candidate_samples[i] for i in top_indices]

def get_few_shot_samples(sampling_method, test_sentence_input, candidate_samples, num_samples_per_file, sbert_model=None, diversity_samples=None):
    if sampling_method == 'random':
        return random.sample(candidate_samples, num_samples_per_file) if len(candidate_samples) > num_samples_per_file else candidate_samples
    elif sampling_method == 'similarity':
        if sbert_model is not None:
            return get_most_similar_samples(test_sentence_input, candidate_samples, num_samples_per_file, sbert_model)
        else:
            print('使用TF-IDF相似度采样（SentenceTransformer加载失败）')
            return get_most_similar_samples_tfidf(test_sentence_input, candidate_samples, num_samples_per_file)
    elif sampling_method == 'diversity':
        few_shot_samples = []
        if diversity_samples is not None:
            for cluster_samples in diversity_samples:
                if len(cluster_samples) > 0:
                    few_shot_samples.append(random.choice(cluster_samples))
        return few_shot_samples
    else:
        raise ValueError('Unknown sampling method!')

def main(
    sampling_method='similarity',
    example_files=None,
    diversity_example_files=None,
    test_file=None,
    output_file='output/gpt_output.jsonl',
    label_semantic_file=None,
    num_samples_per_file=2,
    label_set=None
):
    if example_files is None or test_file is None or label_semantic_file is None or diversity_example_files is None or label_set is None:
        raise ValueError('example_files, test_file, label_semantic_file, diversity_example_files, label_set 不能为空，请检查数据集映射！')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(label_semantic_file, 'r', encoding='utf-8') as file:
        label_semantic = file.read().strip()
    
    diversity_samples = []
    if sampling_method == 'diversity':
        for file in diversity_example_files:
            if os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    samples = [json.loads(line.strip()) for line in f if line.strip()]
                    diversity_samples.append(samples)
            else:
                print(f"Warning: Diversity example file {file} not found. Skipping.")
    
    sbert_model = None
    if sampling_method == 'similarity':
        sbert_model = try_load_sentence_transformer("/mnt/data0/PLMs/all-MiniLM-L6-v2-main/")
        if sbert_model is None:
            print('本地all-MiniLM-L6-v2加载失败，尝试在线加载...')
            sbert_model = try_load_sentence_transformer('sentence-transformers/all-MiniLM-L6-v2')
        if sbert_model is None:
            print('所有SentenceTransformer模型加载失败，将使用TF-IDF兜底。')
    
    candidate_samples = []
    for file in example_files:
        if os.path.exists(file):
            candidate_samples.extend(read_random_samples(file, 1000))
        else:
            print(f"Warning: Example file {file} not found. Skipping.")
    
    print(f"成功加载 {len(candidate_samples)} 个候选样本")
    
    with open(test_file, 'r', encoding='utf-8') as test_f:
        test_lines = test_f.readlines()
    label_set_str = ', '.join([f"'{l}'" for l in label_set])
    with open(output_file, 'w', encoding='utf-8') as output_f:
        for test_line in tqdm(test_lines, desc=f"Processing with GPT using {sampling_method} sampling"):
            try:
                test_sample = json.loads(test_line)
                test_sentence_input = test_sample.get('sentence', test_sample.get('text', ''))
                
                few_shot_samples = get_few_shot_samples(
                    sampling_method,
                    test_sentence_input,
                    candidate_samples,
                    num_samples_per_file,
                    sbert_model=sbert_model,
                    diversity_samples=diversity_samples
                )
                
                gold_entities = []
                if 'entities' in test_sample:
                    for entity in test_sample['entities']:
                        gold_entities.append({"entity": entity["entity"], "type": entity["type"]})
                
                task_description = (
                    f"I am an excellent linguist. Given entity label set: [{label_set_str}] and label semantic descriptions and diverse reference demonstrations. "
                    "The task is to recognize the named entities in the given text as below examples do. Ensure not to split entity into multiple parts and pay attention to identifying nested structures in sentence."
                )
                
                reasoning = ""
                for sample_data in few_shot_samples:
                    sentence_input = sample_data.get('sentence', sample_data.get('text', ''))
                    clues = None
                    if 'inference' in sample_data:
                        clues = sample_data['inference']
                    elif 'clues' in sample_data:
                        clues = sample_data['clues']
                    elif 'reasoning' in sample_data:
                        clues = sample_data['reasoning']
                    
                    if clues:
                        reasoning += f"Text: {sentence_input}\nInference Output: \n{clues}\n"
                
                output_limitation = (
                    f"Follow the above reasoning chain to generate the complete inference process, ensuring that only [{label_set_str}] entities are produced and no other entity types are generated. "
                    "Strictly follow the format: 1. CLUES 2. REASONING 3. ENTITIES"
                )
                
                prompt = (
                    "Task Description: \n" + task_description +
                    "\nLabel Semantic Description: \n" + label_semantic +
                    "\nFew Shot Demonstrations:\n" + reasoning +
                    output_limitation +
                    "\nTest Input:\n" + test_sentence_input
                )
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # 可以根据需要调整模型
                    messages=[
                        {"role": "system", "content": "You are an excellent linguist. Please recognize the named entities in the given text and provide inference and reasoning."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=1024,
                )
                
                model_output = response.choices[0].message.content
                
                pred_entities = []
                entities_section_start = model_output.find('3. ENTITIES')
                if entities_section_start != -1:
                    entities_section = model_output[entities_section_start:]
                    entities_data = entities_section.split("3. ENTITIES", 1)[1].strip()
                    
                    matches = re.findall(r'\{.*?\}', entities_data, re.DOTALL)
                    for match in matches:
                        try:
                            json_line = match.replace("'", '"')
                            entity_info = json.loads(json_line)
                            pred_entities.append(entity_info)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing entity: {match}. Error: {e}")
                
                output_data = {
                    "text": test_sentence_input,
                    "gold_entities": gold_entities,
                    "pred_entities": pred_entities,
                    "inference_output": model_output
                }
                output_f.write(json.dumps(output_data) + "\n")
            
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue
    
    print(f"推理结果已保存到 {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT NER Few-shot Inference')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称,如ncbi')
    parser.add_argument('--sampling_method', type=str, required=True, choices=['random', 'similarity', 'diversity'], help='采样方式')
    parser.add_argument('--num_samples_per_file', type=int, default=None, help='每个文件采样数量（默认等于当前数据集簇数）')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='使用的GPT模型')
    args = parser.parse_args()
    
    if args.dataset not in DATASET_MAP:
        raise ValueError(f'未知数据集: {args.dataset}，请在DATASET_MAP中添加！')
    
    dataset_info = DATASET_MAP[args.dataset]
    num_samples_per_file = args.num_samples_per_file
    
    if num_samples_per_file is None:
        num_samples_per_file = len(dataset_info['diversity_example_files'])
    
    output_file = os.path.join(
        dataset_info['output_dir'],
        f'gpt_{args.sampling_method}.jsonl'
    )
    
    main(
        sampling_method=args.sampling_method,
        example_files=dataset_info['example_files'],
        diversity_example_files=dataset_info['diversity_example_files'],
        test_file=dataset_info['test_file'],
        output_file=output_file,
        label_semantic_file=dataset_info['label_semantic_file'],
        num_samples_per_file=num_samples_per_file,
        label_set=dataset_info['label_set']
    )
