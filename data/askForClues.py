import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== ENV ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

# ========== FEW-SHOT EXAMPLES ==========
FEW_SHOT_EXAMPLES = [
    {
        "text": "This was in contrast to normal peripheral blood B lymphocytes or CD5+) B cells isolated from tonsils, in which this phosphorylation was absent.",
        "entities": [
            {"entity": "CD5+) B cells", "type": "cell_type"},
            {"entity": "B lymphocytes", "type": "cell_type"}
        ],
        "inference": (
            "1. CLUES:\n"
            "- 'B lymphocytes' are a type of white blood cell.\n"
            "- 'CD5+) B cells' refers to a specific subset of B lymphocytes expressing the CD5 marker.\n"
            "- 'peripheral blood' and 'tonsils' are locations.\n"
            "- 'phosphorylation' is a biological process.\n\n"
            "2. REASONING:\n"
            "- 'B lymphocytes' are immune cells, hence labeled as 'cell_type'.\n"
            "- 'CD5+) B cells' are a subtype of B lymphocytes, also labeled as 'cell_type'.\n\n"
            "3. ENTITIES:\n"
            "{'entity': 'CD5+) B cells', 'type': 'cell_type'}\n"
            "{'entity': 'B lymphocytes', 'type': 'cell_type'}"
        )
    }
]

# ========== PROMPT CONSTRUCTION ==========
def build_prompt(text, entities):
    prompt = ""
    for example in FEW_SHOT_EXAMPLES:
        example_entities = ", ".join([f"{{'entity': '{e['entity']}', 'type': '{e['type']}'}}" for e in example["entities"]])
        prompt += f"Text: {example['text']}\n"
        prompt += f"Entities: {example_entities}\n"
        prompt += f"Inference:\n{example['inference']}\n\n"

    current_entities = ", ".join([f"{{'entity': '{e['entity']}', 'type': '{e['type']}'}}" for e in entities])
    prompt += f"Text: {text}\n"
    
    prompt += f"Entities: {current_entities}\n"
    prompt += (
    "Please generate an inference in the following structure:\n"
    "Important: Your inference must be entirely based on the provided sentence and entities.\n"
    "1. CLUES: Identify key information based on the entities and context.\n"
    "2. REASONING: Explain the reasoning from input to output.\n"
    "3. ENTITIES: List all inferred entities in the format {'entity': 'xxx', 'type': 'xxx'}.\n\n"
    # "Important: Your inference must be entirely based on the provided sentence and entities 'ADR'.\n"
    # "必须按照当前推理的句子严格生成推理链,如果当前推理句子的Entities为空，不要凭空想象制造不存在的实体！！\n\n"
    "Important: You MUST NOT hallucinate or invent entities. If no entities are provided, your reasoning must reflect that and leave ENTITIES empty."
    # "Return format:\n"
    # "1. CLUES:\n...\n\n2. REASONING:\n...\n\n3. ENTITIES:\n..."
    )
    return prompt

# ========== MODEL LOADING ==========
def load_model_tokenizer(model_path):
    print("Loading model and tokenizer from:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    print("Model loaded.")
    return model, tokenizer

# ========== GENERATION ==========
def run_inference(model, tokenizer, prompt, max_new_tokens=2048):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            # **inputs,
            # max_new_tokens=max_new_tokens,
            # do_sample=True,
            # temperature=0.0,
            # top_p=0.9,
            # num_beams=1,
            # top_k=50,
            # eos_token_id=tokenizer.eos_token_id,
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # 不采样
            num_beams=1,               # 单路径生成
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens (excluding the prompt)
    generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

# ========== MAIN FUNCTION ==========
def generate_inference_file(input_path, output_path, model, tokenizer):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Generating Inference", dynamic_ncols=True):
            try:
                sample = json.loads(line.strip())
                text = sample["sentence"]
                entities = sample["entities"]
                prompt = build_prompt(text, entities)

                inference = run_inference(model, tokenizer, prompt)
                # 生成推理结果后，马上打印
                print("="*30)
                print("Prompt:\n", prompt)
                print("Inference:\n", inference)
                print("="*30)

                sample["inference"] = inference
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"[Error] Skipping due to: {e}")
                continue

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    MODEL_PATH = "/mnt/data0/PLMs/Falcon3-7B-Instruct/"
    INPUT_PATH = "/mnt/data0/LSDNER/data/ncbi/testtest.jsonl"
    OUTPUT_PATH = "/mnt/data0/LSDNER/data/ncbi/testtest1.jsonl"

    model, tokenizer = load_model_tokenizer(MODEL_PATH)
    generate_inference_file(INPUT_PATH, OUTPUT_PATH, model, tokenizer)
