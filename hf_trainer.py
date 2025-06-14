import os
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.preprocessing import normalize
import numpy as np

query_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
queries = [
    "Ask a question about software stack used to build this website.",
    "Ask a question about the projects and technologies that Andrew has worked on.",
    "Ask a question about Andrew's experience with web development.",
    "Ask a question about Andrew's educational background.",
    "Ask a question about Andrew's work experience."
]
queries = [query_prompt + query for query in queries]
# docs do not need any prompts
docs = load_dataset("json", data_files="dataset/cv_dataset.json", split="train", streaming=True)

# The path of your model after cloning it
model_dir = "NovaSearch/stella_en_400M_v5"

vector_dim = 1024
vector_linear_directory = f"2_Dense_{vector_dim}"
# you can also use this model without the features of `use_memory_efficient_attention` and `unpad_inputs`. It can be worked in CPU.
# model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,use_memory_efficient_attention=False,unpad_inputs=False).cuda().eval()
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

from huggingface_hub import hf_hub_download
vector_linear_path = hf_hub_download(
    repo_id=model_dir,
    filename=f"{vector_linear_directory}/pytorch_model.bin",
    repo_type="model"
)
vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim).cuda()
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    torch.load(vector_linear_path).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.cuda()

# Embed the queries
with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_data = tokenizer(queries, padding="longest", truncation=True, max_length=512, return_tensors="pt").to(device)
    input_data = {k: v.cuda() for k, v in input_data.items()}
    attention_mask = input_data["attention_mask"]
    last_hidden_state = model(**input_data)[0]
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    query_vectors = normalize(vector_linear(query_vectors).cpu().numpy())

# Embed the documents
with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_data = tokenizer([query_prompt], padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    input_data = {k: v.cuda() for k, v in input_data.items()}
    attention_mask = input_data["attention_mask"]
    last_hidden_state = model(**input_data)[0]
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    docs_vectors = normalize(vector_linear(docs_vectors).cpu().numpy())

np.save("query_vectors.npy", query_vectors)
np.save("docs_vectors.npy", docs_vectors)
