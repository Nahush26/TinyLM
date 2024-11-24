import json
import torch
import numpy as np
from utils import get_probs, save_tensors
from utils import generate_greedy_response
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the JSON file
with open("xsum_gen_4.json", "r") as f:
    results = json.load(f)

# Load a different model and tokenizer
new_model_name = "google/gemma-2-2b"  # Replace with your desired model
tokenizer = AutoTokenizer.from_pretrained(new_model_name)
model = torch.load("gemma-2-2b-it_0.1_depth.pt")
model.config.kv_heads = None
model.eval()

# Function to calculate log-likelihood
def calculate_log_likelihood(prompts):
    # Tokenize the prompt and generated tokens
    input_ids = tokenizer.encode(prompts, return_tensors="pt").to(device)
    # Pass through the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        logits = outputs.logits

    # Shift the logits and targets to calculate the loss


    # Compute log softmax and likelihood
    log_probs = softmax(logits, dim=-1)
    [print(torch.sum(log_probs))]

    return log_probs
loaded = np.load("teacher_probs_4.npz")

# Access individual tensors
for key in loaded.files:
    tensor = torch.tensor(loaded[key]) 
    print(tensor.shape)
# exit()
# Iterate over the samples and calculate log likelihoods
log_likelihoods = []
tensor_list = []
for i, sample in enumerate(results):

    prompt = sample["prompt"]
    generated_response = sample["generated_response"]
    length = tokenizer.encode(prompt, return_tensors="pt").shape[1]
    generated_tokens = tokenizer.encode(generated_response, return_tensors="pt").shape[1]

    # generated_tokens = tokenizer.encode(generated_response)
     # Tokenize response
    probs = get_probs(model, generated_response, tokenizer, device, generate = False, prev_length = length)
    # Reconstruct the prompt
    # prompt = f"Summarise the below document in a single line  {question}"
    
    print("before length ", length)
    print("after length", generated_tokens)
    tensor_list.append(probs)
    print(probs.shape)
    # exit()
    # Calculate log likelihood
    # probs= calculate_log_likelihood(generated_response)
    # to_check = generated_tokens[length:]
    # probs = probs[0, length+1:, :].cpu().numpy()
    # print("diff", len(to_check))
    # teacher_rank = []
    # frac_5 = 0
    # frac_10 = 0
    # frac_50 = 0 
    # frac_1p = 0
    # for j in range(len(to_check)):
    #     T = list(np.argsort(logs[j])[::-1])
    #     teacher_rank.append(T.index(to_check[j]))
    #     top_5_indexes = np.argsort(logs[j])[-5:][::-1]
    #     top_10_indexes = np.argsort(logs[j])[-10:][::-1]
    #     top_50_indexes = np.argsort(logs[j])[-50:][::-1]
    #     top_1p_indexes = np.argsort(logs[j])[-2560:][::-1]
    #     if to_check[j] in top_5_indexes:
    #          frac_5+=1
    #     if to_check[j] in top_10_indexes:
    #          frac_10+=1
    #     if to_check[j] in top_50_indexes:
    #          frac_50+=1
    #     if to_check[j] in top_1p_indexes:
    #         frac_1p+=1
    # frac_5/=len(to_check)
    # frac_10/=len(to_check)
    # frac_50/=len(to_check)
    # frac_1p/=len(to_check)
        
    # print(teacher_rank)
    # log_likelihoods.append({"id": sample["id"], "frac_5": frac_5,"frac_10": frac_10,"frac_50": frac_50,"frac_1p" : frac_1p, "teacher rank" : teacher_rank})
    
    # Print progress
    if i % 10 == 0:
        print(f"Processed {i} examples...")

save_tensors(tensor_list, "student_probs_4.npz")
# Save the log-likelihoods to a new JSON file
# with open("log_likelihoods_xsu.json", "w") as f:
#     json.dump(log_likelihoods, f, indent=4)

# print("Log likelihood calculation complete. Results saved to log_likelihoods.json.")
