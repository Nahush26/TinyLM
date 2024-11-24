from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
import numpy as np
import gc

def generate_greedy_response(model, tokenizer, prompt, device, max_length=8000):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length=max_length, do_sample=False, temperature=0.0,return_dict_in_generate=True, output_scores=True)
    generated_tokens = output_ids.sequences[0].tolist()
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response, generated_tokens

def get_probs(model, input, tokenizer, device, generate = True, max_length = 8000, prev_length = None):
    input_ids = tokenizer.encode(input, return_tensors="pt").to(device)
    if generate : 
        output_ids = model.generate(input_ids, max_length=max_length, do_sample=False, temperature=0.0,return_dict_in_generate=True, output_scores=True)
        logits = output_ids.scores
        cpu_tensors = [tensor.cpu() for tensor in logits]
        probs = torch.cat(cpu_tensors, dim=0)
        probs = softmax(probs, dim=-1)
        probs = probs.numpy()
        del output_ids
        
    else:
        outputs = model(input_ids=input_ids, labels=input_ids)
        probs = softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        probs = probs[0, prev_length-1:, :]
        del outputs
    torch.cuda.empty_cache()
    gc.collect()
    return probs

def save_tensors(tensor_list, path):
    numpy_arrays = {f"tensor_{i}": tensor for i, tensor in enumerate(tensor_list)}
    np.savez(path, **numpy_arrays)
    print("Tensors Saved")

def compute_average_kld(tensor1, tensor2):

    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape."

    # Assertion: Each row in both tensors must sum to 1 (valid probability distributions)
    # assert torch.allclose(tensor1.sum(dim=1), torch.ones(tensor1.size(0), device=tensor1.device)), \
    #     "Each row in tensor1 must sum to 1 (valid probability distribution)."
    # assert torch.allclose(tensor2.sum(dim=1), torch.ones(tensor2.size(0), device=tensor2.device)), \
    #     "Each row in tensor2 must sum to 1 (valid probability distribution)."

    # Compute KLD for each row (F.kl_div expects log probabilities for the first input)
    kld = F.kl_div(torch.log(tensor1), tensor2, reduction="none").sum(dim=1) ## tensor2*(log(tensor1)- log(tensor2))
    
    # Average the KLD across samples
    average_kld = kld.mean().item()

    return average_kld


def cov(m):
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.shape[-1] - 1)  # 1 / N
    m -= torch.mean(m, dim=(1, 2), keepdim=True)
    mt = torch.transpose(m, 1, 2)  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def compute_rank_correlation(x, y):
    x, y = rankmin(x), rankmin(y)
    return corrcoef(x, y)


def corrcoef(x, y):
    batch_size = x.shape[0]
    x = torch.stack((x, y), 1)
    # calculate covariance matrix of rows
    c = cov(x)
    # normalize covariance matrix
    d = torch.diagonal(c, dim1=1, dim2=2)
    stddev = torch.pow(d, 0.5)
    stddev = stddev.repeat(1, 2).view(batch_size, 2, 2)
    c = c.div(stddev)
    c = c.div(torch.transpose(stddev, 1, 2))
    return c[:, 1, 0]
import numpy as np

def spearman_rank_correlation(array1, array2, weight = None):
    """
    Compute the Spearman rank correlation coefficient between two arrays.

    Args:
        array1 (numpy.ndarray): First array of shape (N,).
        array2 (numpy.ndarray): Second array of shape (N,).

    Returns:
        float: Spearman rank correlation coefficient.
    """
    assert array1.shape == array2.shape, "Arrays must have the same shape."
    N = array1.shape[0]

    # Step 1: Compute ranks
    rank1 = np.argsort(np.argsort(array1))
    rank2 = np.argsort(np.argsort(array2))
    # print(rank1, rank2)

    # Step 2: Compute rank differences
    d = rank1 - rank2
    if weight is not None:
        d*=weight

    # Step 3: Compute Spearman rank correlation using the formula
    spearman_corr = 1 - (6 * np.sum(d**2)) / (N * (N**2 - 1))

    return spearman_corr


def compute_spr_seq(array1, array2):
    array1, array2 = array1.numpy(), array2.numpy()
    N = array1.shape[0]
    spr = 0
    for i in range(N):
        weight = array1[i]/np.max(array1[i])
        # print(np.max(weight))
        spr_i = spearman_rank_correlation(array1[i], array2[i])
        # print(spr_i)
        spr+=spr_i
    print("NET", spr/N)



