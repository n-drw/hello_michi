#!/usr/bin/env python3
"""
RAG setup using Google Gemma (non-reasoning) + Stella embeddings
This replaces DeepSeek R1 (reasoning model) with Gemma for cleaner chatbot responses
"""
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import faiss
import json

# Model configuration
LANGUAGE_MODEL = "google/gemma-2-2b-it"  # Non-reasoning instruction-tuned model
EMBEDDING_MODEL = "NovaSearch/stella_en_400M_v5"  # Same embedding model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load language model (Gemma - non-reasoning)
print(f"\n📥 Loading language model: {LANGUAGE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    LANGUAGE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL, trust_remote_code=True)

# Load embedding model (Stella)
print(f"📥 Loading embedding model: {EMBEDDING_MODEL}")
emb_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL,
    trust_remote_code=True
).to(device).eval()
emb_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)

# Load dataset
print("\n📄 Loading datasets...")
with open("dataset/cv_dataset.json", "r") as f:
    cv_data = json.load(f)

with open("dataset/qa_dataset.json", "r") as f:
    qa_data = json.load(f)

# Prepare documents from CV
documents = []
metadata = []

# Personal info
personal = cv_data["cv_data"]["personal_info"]
doc = f"Personal Info: {personal['name']} is based in {personal['contact']['location']}. {personal['summary']}"
documents.append(doc)
metadata.append({"type": "personal", "source": "cv"})

# Experience
for job in cv_data["cv_data"]["experience"]:
    job_desc = " ".join(job["description"])
    doc = f"Experience: {job['role']} at {job['company']} ({job['dates']}). {job_desc}"
    documents.append(doc)
    metadata.append({"type": "experience", "company": job["company"], "source": "cv"})

# Skills
for skill_group in cv_data["cv_data"]["skills"]:
    doc = f"Skills in {skill_group['category']}: " + ", ".join(skill_group["technologies"])
    documents.append(doc)
    metadata.append({"type": "skills", "category": skill_group["category"], "source": "cv"})

# Education
for edu in cv_data["cv_data"]["education"]:
    doc = f"Education: {edu['degree']} from {edu['institution']} ({edu['dates']}) in {edu['field']}"
    documents.append(doc)
    metadata.append({"type": "education", "source": "cv"})

# Add Q&A pairs as documents
for qa in qa_data["qa_pairs"]:
    doc = f"Q: {qa['question']}\nA: {qa['answer']}"
    documents.append(doc)
    metadata.append({"type": "qa", "question": qa["question"], "source": "qa"})

print(f"✅ Loaded {len(documents)} documents")

# Function to encode text into embeddings
def encode_text(text):
    inputs = emb_tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = emb_model(**inputs)
    
    # Use mean pooling to get a single vector
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Create embeddings for all documents
print("\n🔢 Creating embeddings...")
doc_embeddings = []
for i, doc in enumerate(documents):
    if i % 10 == 0:
        print(f"  Progress: {i}/{len(documents)}")
    emb = encode_text(doc)
    doc_embeddings.append(emb)

print(f"✅ Created {len(doc_embeddings)} embeddings")

# Build FAISS index
embedding_dim = emb_model.config.hidden_size
print(f"\n🗂️  Building FAISS index (dim={embedding_dim})")
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.vstack(doc_embeddings))
print(f"✅ Index built with {index.ntotal} vectors")

def query_chatbot(user_query, k=3, max_new_tokens=256):
    """
    Query the RAG chatbot with Gemma + Stella
    
    Args:
        user_query: User's question
        k: Number of relevant documents to retrieve
        max_new_tokens: Max tokens to generate
    
    Returns:
        Generated response
    """
    # Retrieve relevant documents
    query_emb = encode_text(user_query)
    D, I = index.search(query_emb, k=k)
    
    # Get top documents
    context_docs = [documents[idx] for idx in I[0]]
    context = "\n\n".join(context_docs)
    
    # Format prompt for Gemma
    # Gemma uses a specific chat template
    prompt = f"""You are Michi, Andrew's helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {user_query}

Answer (respond naturally and concisely as Michi):"""
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the answer part (after the prompt)
    answer = full_response[len(prompt):].strip()
    
    return answer

# Test queries
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing RAG Chatbot with Gemma + Stella")
    print("="*60 + "\n")
    
    test_queries = [
        "What projects have you worked on?",
        "What is your educational background?",
        "What are your technical skills?",
        "Tell me about your experience with AI",
        "What programming languages do you know?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 60)
        response = query_chatbot(query)
        print(f"💬 Response: {response}")
        print()
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = query_chatbot(user_input)
        print(f"Michi: {response}\n")
