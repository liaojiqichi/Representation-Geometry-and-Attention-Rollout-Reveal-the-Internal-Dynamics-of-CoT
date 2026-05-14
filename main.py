import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 200
TOP_K = 50

print(f"Loading {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    output_attentions=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()
NUM_LAYERS = model.config.num_hidden_layers

PREFIX_ALGEBRA = "Solve step by step using algebra. Let x be unknown. "
PREFIX_ARITHMETIC = "Solve step by step using arithmetic only. "

FINAL_PHRASE = "\nFinal answer: "

def generate_cot(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def topk_tvd(p, q, k=50):
    idx = np.argsort(p)[-k:]
    p_k = p[idx]
    q_k = q[idx]
    return 0.5 * np.sum(np.abs(p_k - q_k))


def dtw_align(seq1, seq2):
    n, m = len(seq1), len(seq2)

    cost = np.full((n+1, m+1), np.inf)
    cost[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            dist = (seq1[i-1] - seq2[j-1]) ** 2
            cost[i, j] = dist + min(
                cost[i-1, j],
                cost[i, j-1],
                cost[i-1, j-1]
            )

    i, j = n, m
    aligned_1 = []
    aligned_2 = []

    while i > 0 and j > 0:
        aligned_1.append(seq1[i-1])
        aligned_2.append(seq2[j-1])

        steps = [
            cost[i-1, j],
            cost[i, j-1],
            cost[i-1, j-1]
        ]
        step = np.argmin(steps)

        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    aligned_1.reverse()
    aligned_2.reverse()

    return np.array(aligned_1), np.array(aligned_2)


def normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    x = x + eps
    s = np.sum(x)
    if s <= 0:
        return np.ones_like(x) / len(x)
    return x / s

def jsd(p, q, eps=1e-8):
    p = normalize(p, eps)
    q = normalize(q, eps)

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return 0.5 * (kl_pm + kl_qm)

def get_attention_and_logits(full_text):
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    layer_attns = []

    for l in range(NUM_LAYERS):
        attn = outputs.attentions[l][0]
        attn = attn.mean(dim=0)
        last = attn[-1, :].cpu().numpy()
        last = normalize(last)
        layer_attns.append(last)

    return probs, layer_attns


def shuffle_cot_tokens(full_text, ctx_len, cot_len):
    tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]

    cot_start = ctx_len
    cot_end = min(ctx_len + cot_len, len(tokens) - 1)

    cot_tokens = tokens[cot_start:cot_end].clone()
    cot_tokens = cot_tokens[torch.randperm(len(cot_tokens))]

    new_tokens = tokens.clone()
    new_tokens[cot_start:cot_end] = cot_tokens

    with torch.no_grad():
        outputs = model(input_ids=new_tokens.unsqueeze(0).to(DEVICE))

    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    return probs

def evaluate_sample(question):
    messages = [{"role": "user", "content": f"{question}\nThink step by step."}]
    base_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    ctx_len = len(tokenizer.encode(base_prompt))

    prompt_A = base_prompt + PREFIX_ALGEBRA
    cot_A = generate_cot(prompt_A)
    full_A = prompt_A + cot_A + FINAL_PHRASE
    cot_len_A = len(tokenizer.encode(PREFIX_ALGEBRA + cot_A))

    probs_A, attn_A = get_attention_and_logits(full_A)
    probs_A_shuffle = shuffle_cot_tokens(full_A, ctx_len, cot_len_A)

    prompt_B = base_prompt + PREFIX_ARITHMETIC
    cot_B = generate_cot(prompt_B)
    full_B = prompt_B + cot_B + FINAL_PHRASE
    cot_len_B = len(tokenizer.encode(PREFIX_ARITHMETIC + cot_B))

    probs_B, attn_B = get_attention_and_logits(full_B)
    probs_B_shuffle = shuffle_cot_tokens(full_B, ctx_len, cot_len_B)

    tvd_base = topk_tvd(probs_A, probs_B, TOP_K)

    tvd_shuffle_A = topk_tvd(probs_A, probs_A_shuffle, TOP_K)
    tvd_shuffle_B = topk_tvd(probs_B, probs_B_shuffle, TOP_K)

    layer_jsd = []

    for l in range(NUM_LAYERS):
        a_raw = attn_A[l][ctx_len:]
        b_raw = attn_B[l][ctx_len:]
        a_aligned, b_aligned = dtw_align(a_raw, b_raw)
        a = normalize(a_aligned)
        b = normalize(b_aligned)

        js = jsd(a, b)
        layer_jsd.append(js)

    return {
        "question": question,
        "tvd_base": tvd_base,
        "tvd_shuffle_A": tvd_shuffle_A,
        "tvd_shuffle_B": tvd_shuffle_B,
        "jsd_last_layer": layer_jsd[-1],
        "jsd_mean": np.mean(layer_jsd),
        "jsd_layerwise": str(layer_jsd)
    }
dataset = load_dataset("gsm8k", "main", split="test")
sampled = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

results = []

for i, item in enumerate(tqdm(sampled)):
    try:
        res = evaluate_sample(item["question"])
        results.append(res)
    except Exception as e:
        print(f"Error {i}: {e}")

df = pd.DataFrame(results)
df.to_csv("fixed_adversarial_results.csv", index=False)

print(f"Samples: {len(df)}")
print(f"TVD base: {df['tvd_base'].mean():.4f}")
print(f"JSD mean: {df['jsd_mean'].mean():.4f}")
print(f"Shuffle impact A: {df['tvd_shuffle_A'].mean():.4f}")
print(f"Shuffle impact B: {df['tvd_shuffle_B'].mean():.4f}")
