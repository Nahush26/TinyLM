from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from utils import get_probs
from utils import generate_greedy_response
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the GSM8k dataset
data = load_dataset('GEM/xsum', split = 'validation')
data_train  = load_dataset('GEM/xsum', split = 'train')
# Load a Hugging Face-compatible model and tokenizer
model_name = "google/gemma-2-2b-it"  # Replace with your LLM model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.kv_heads = None

def create_fewshot(num_shots = 3):
    prompt = ""
    for i in range(num_shots):
        prompt+= f"Document : {data_train[i]['document']}\nSummary : {data_train[i]['target']}\n"
    return prompt


# List to store all generated tokens
all_generated_tokens = []

results = []
tensor_list = []
for i, example in enumerate(data):
    if(i>20):
        break
    doc = example["document"]
    prompt = create_fewshot(4)
    prompt+=f"Document : {doc}\nSummary :"

    probs = get_probs(model, prompt, tokenizer, device)

    response, _ = generate_greedy_response(model, tokenizer, prompt, device)

    tensor_list.append(probs)
    
    # Append generated tokens to the list
    # all_generated_tokens.append(tokens)
    
    # # Store result
    results.append({"id" : i, "prompt": prompt, "generated_response": response})
    
    # Print progress
    if i % 10 == 0:
        print(f"Processed {i} examples...")

# Save results
numpy_arrays = {f"tensor_{i}": tensor for i, tensor in enumerate(tensor_list)}
np.savez("teacher_probs_4.npz", **numpy_arrays)

import json
with open("xsum_gen_4.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Evaluation complete. Results saved.")
