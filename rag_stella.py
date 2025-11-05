from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import faiss
from datasets import load_dataset
import json

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
em_model_id = "NovaSearch/stella_en_400m_v5"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

emb_model = AutoModel.from_pretrained(em_model_id, trust_remote_code=True).to(device).eval()
emb_tokenizer = AutoTokenizer.from_pretrained(em_model_id, trust_remote_code=True)

with open("dataset/cv_dataset.json", "r") as f:
    data = json.load(f)

documents = []
personal = data["cv_data"]["personal_info"]
documents.append(f"Personal Info: {personal['name']} is based in {personal['contact']['location']}. {personal['summary']}")

# Experience
for job in data["cv_data"]["experience"]:
    job_desc = " ".join(job["description"])
    documents.append(f"Experience: {job['role']} at {job['company']} ({job['dates']}). {job_desc}")

# Skills
for skill_group in data["cv_data"]["skills"]:
    skill_text = f"Skills in {skill_group['category']}: " + ", ".join(skill_group["technologies"])
    documents.append(skill_text)

# Education
for edu in data["cv_data"]["education"]:
    documents.append(f"Education: {edu['degree']} from {edu['institution']} ({edu['dates']}) in {edu['field']}")

# Function to encode text into embeddings
def encode_text(text):
    inputs = emb_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    # Use mean pooling to get a single vector
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Create embeddings for all documents
doc_embeddings = []
for doc in documents:
    emb = encode_text(doc)
    doc_embeddings.append(emb)

embedding_dim = emb_model.config.hidden_size  # Get the correct embedding dimension from the model
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.vstack(doc_embeddings))  # Stack all embeddings vertically

def query_chatbot(user_query):
    query_emb = encode_text(user_query)
    D, I = index.search(query_emb, k=1)
    top_doc = documents[I[0][0]]
    inputs = tokenizer(f"Query: {user_query}\nContext: {top_doc}", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=4096)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(query_chatbot("What projects have you worked on?"))
print(query_chatbot("What is your educational background?"))
print(query_chatbot("What are your skills?"))