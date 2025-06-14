import os
from datasets import load_dataset
import numpy as np
import torch
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize

def main():
    VECTOR_DIM = 1024
    dataset = load_dataset("json", data_files="dataset/cv_dataset.json", split="train", streaming=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("NovaSearch/stella_en_400M_v5", trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("NovaSearch/stella_en_400M_v5", trust_remote_code=True)

    vector_linear = torch.nn.Linear(1024, VECTOR_DIM).to(device)

    with torch.no_grad():
        doc_inputs = tokenizer(dataset, padding=True, truncation=True, return_tensors="pt", max_length=512)
        doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
        doc_outputs = model(**doc_inputs)
        last_hidden = doc_outputs.last_hidden_state
        attention_mask = doc_inputs['attention_mask'].unsqueeze(-1)
        doc_vecs = torch.sum(last_hidden * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        doc_vecs = vector_linear(doc_vecs).cpu().numpy()
        doc_vecs = normalize(doc_vecs)

    # Save the vectors to a file
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        # Process query
        query_prompt = "Instruct: Retrieve relevant CV info. Query: " + query
        with torch.no_grad():
            q_input = tokenizer([query_prompt], padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            q_outputs = model(**q_input)
            q_last_hidden = q_outputs.last_hidden_state
            q_mask = q_input['attention_mask'].unsqueeze(-1)
            q_vec = torch.sum(q_last_hidden * q_mask, dim=1) / torch.clamp(q_mask.sum(dim=1), min=1e-9)
            q_vec = vector_linear(q_vec).cpu().numpy()
            q_vec = normalize(q_vec)
        
        # Compute similarities
        similarities = q_vec @ doc_vecs.T
        top_idx = similarities.argmax()
        top_score = similarities[0][top_idx]
        top_doc = dataset[top_idx]
    
    if __name__ == "__main__":
        main()