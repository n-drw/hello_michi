import json
import os
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

# Import for custom embeddings
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Transformers imports for the LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering, pipeline
from langchain_core.language_models import BaseLLM
from langchain.chains import RetrievalQA

# PEFT (Parameter-Efficient Fine-Tuning) imports for LoRA
from peft import PeftModel, PeftConfig
import torch
import numpy as np

# Define constants and paths
CHROMA_DB_DIR = "./chroma_db"
BASE_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Base model
LORA_EMBEDDING_MODEL = "./lora-sentence-transformer"      # Local LoRA adapter path
CV_DATASET_PATH = "./dataset/cv_dataset.json"
QA_DATASET_PATH = "./dataset/qa_dataset.json"
COLLECTION_NAME = "cv_qa_collection"

# LLM constants
LLM_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
LOCAL_MODEL_DIR = "./models/smollm2-1.7b"

# Custom Embeddings class to fix the HuggingFaceEmbeddings bug
class CustomEmbeddings(Embeddings):
    """Custom embeddings class that uses sentence_transformers directly.
    This avoids the 'dict' object has no attribute 'replace' bug in LangChain's HuggingFaceEmbeddings.
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        # Convert documents to embeddings
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        # Convert query to embedding
        return self.model.encode(query).tolist()

def load_datasets():
    """Load CV and QA datasets from JSON files"""
    with open(CV_DATASET_PATH, 'r') as f:
        cv_data = json.load(f)
    
    with open(QA_DATASET_PATH, 'r') as f:
        qa_data = json.load(f)
    
    return cv_data, qa_data

def prepare_documents_from_cv(cv_data: Dict[str, Any]) -> List[Document]:
    """Prepare documents from CV data for indexing in ChromaDB"""
    documents = []
    
    for info in cv_data["cv_data"]["personal_info"]:
        doc = Document(
            page_content=f"Name: {info['name']}\nContact: {info['contact']['email']}, {info['contact']['location']}\n"
                        f"Languages: {', '.join(info['languages'])}\nSummary: {info['summary']}",
            metadata={"source": "personal_info", "type": "bio"}
        )
        documents.append(doc)
    
    for exp in cv_data["cv_data"]["experience"]:
        description = "\n".join(exp["description"])
        doc = Document(
            page_content=f"Role: {exp['role']}\nCompany: {exp['company']}\nDates: {exp['dates']}\n"
                        f"Location: {exp['location']}\nDescription: {description}\n"
                        f"Tags: {', '.join(exp['tags'])}",
            metadata={"source": "experience", "company": exp["company"], "role": exp["role"]}
        )
        documents.append(doc)
    
    for skill in cv_data["cv_data"]["skills"]:
        doc = Document(
            page_content=f"Category: {skill['category']}\n"
                        f"Technologies: {', '.join(skill['technologies'])}\n"
                        f"Tags: {', '.join(skill['tags'])}",
            metadata={"source": "skills", "category": skill["category"]}
        )
        documents.append(doc)
    
    for edu in cv_data["cv_data"]["education"]:
        doc = Document(
            page_content=f"Institution: {edu['institution']}\nDegree: {edu['degree']}\n"
                        f"Field: {edu['field']}\nDates: {edu['dates']}",
            metadata={"source": "education", "institution": edu["institution"]}
        )
        documents.append(doc)
    
    if "projects" in cv_data["cv_data"]:
        for project in cv_data["cv_data"]["projects"]:
            doc = Document(
                page_content=f"Name: {project['name']}\nDescription: {project['description']}\n"
                            f"Technologies: {', '.join(project['technologies'])}",
                metadata={"source": "projects", "name": project["name"]}
            )
            documents.append(doc)
    
    return documents

def prepare_qa_pairs(qa_data: List[Dict[str, str]]) -> List[Document]:
    """Prepare QA pairs as documents for indexing in ChromaDB"""
    documents = []
    
    for i, qa_pair in enumerate(qa_data):
        doc = Document(
            page_content=f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}",
            metadata={"source": "qa_dataset", "id": i, "question": qa_pair["question"]}
        )
        documents.append(doc)
    
    return documents

def load_lora_qa_model():
    """Load the LoRA QA model for answering questions"""
    try:
        print("Loading LoRA adapter for QA...")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_EMBEDDING_MODEL)
        base_model = AutoModelForQuestionAnswering.from_pretrained(BASE_EMBEDDING_MODEL)
        
        peft_config = PeftConfig.from_pretrained(LORA_EMBEDDING_MODEL)
        model = PeftModel.from_pretrained(base_model, LORA_EMBEDDING_MODEL)
        model.eval()  # Set to evaluation mode
        
        print(f"Successfully loaded LoRA adapter from {LORA_EMBEDDING_MODEL}")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        print("Could not load LoRA QA model")
        return None, None

def create_embedding_function():
    """Create a standard embedding function using the base embedding model"""
    print(f"Using standard embedding function with model {BASE_EMBEDDING_MODEL}")
    return CustomEmbeddings(model_name=BASE_EMBEDDING_MODEL)

def setup_chroma_client():
    """Set up and return a ChromaDB client with the specified embedding function"""
    embedding_function = create_embedding_function()
    
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        collection._embedding_function = embedding_function
        print(f"Retrieved existing collection: {COLLECTION_NAME}")
    except ValueError:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"description": "Resume and QA data for Andrew Cabrera"}
        )
        print(f"Created new collection: {COLLECTION_NAME}")
    
    return client, collection

def add_documents_to_collection(collection, documents: List[Document], batch_size=100):
    """Add documents to the ChromaDB collection in batches"""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    print(f"Added {len(documents)} documents to collection")

def setup_langchain_retriever(collection, k=4):
    """Set up a LangChain retriever using ChromaDB collection"""
    embeddings = CustomEmbeddings(model_name=BASE_EMBEDDING_MODEL)
    
    chroma_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    return retriever
    
    embeddings = CustomLoRAEmbeddings()
    
    chroma_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    return retriever

def setup_llm_model():
    """Setup and load the DeepSeek-R1-Distill-Qwen-1.5B model"""
    print("Setting up LLM model...")
    
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=LOCAL_MODEL_DIR
        )
        
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True
        )
        
        llm = HuggingFacePipeline(pipeline=text_generator)
        
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to simple RAG chain...")
        return None

def create_lora_qa_chain(retriever: BaseRetriever):
    """Create a RAG chain that uses the LoRA QA model to answer questions from retrieved documents"""
    lora_model, tokenizer = load_lora_qa_model()
    
    template = """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    Answer: """
    
    qa_prompt = PromptTemplate.from_template(template)
    
    if lora_model is not None and tokenizer is not None:
        def lora_qa_function(inputs):
            question = inputs["question"]
            context_docs = inputs["context"]
            
            for doc in context_docs:
                if "Question: " + question in doc.page_content:
            for doc in context_docs:
                if "Question: " + question in doc.page_content:
                    return doc.page_content.split("Answer: ")[1].strip()
            
            try:
                context = "\n\n".join([doc.page_content for doc in context_docs])
                
                inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True, max_length=512)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = lora_model(**inputs)
                
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1
                
                if answer_end <= answer_start:
                    answer_end = answer_start + 1
                
                answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
                answer = tokenizer.decode(answer_tokens)
                
                answer = answer.strip()
                if not answer or len(answer) < 3:
                    return extract_answer_from_docs(question, context_docs)
                
                return answer
            except Exception as e:
                print(f"Error using LoRA QA model: {e}")
                return extract_answer_from_docs(question, context_docs)
        
        rag_chain = (
            {"context": retriever, "question": lambda x: x["question"]}
            | qa_prompt
            | lora_qa_function
        )
    else:
        rag_chain = create_simple_extraction_chain(retriever)
    
    return rag_chain

def extract_answer_from_docs(question, context_docs):
    for doc in context_docs:
        if "question" in doc.metadata and doc.metadata["question"] == question:
            if "Answer: " in doc.page_content:
                return doc.page_content.split("Answer: ")[1].strip()
    
    answer_parts = []
    for doc in context_docs:
        if "Answer: " in doc.page_content:
            answer_parts.append(doc.page_content.split("Answer: ")[1].strip())
    
    if answer_parts:
        return "\n".join(answer_parts)
    
    return "I couldn't find a specific answer to that question in the provided documents."

def create_simple_extraction_chain(retriever: BaseRetriever):
    """Create a simple RAG chain that extracts answers from retrieved documents"""
    template = """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    Answer: """
    
    qa_prompt = PromptTemplate.from_template(template)
    
    def simple_answer_function(inputs):
        question = inputs["question"]
        context = inputs["context"]
        return extract_answer_from_docs(question, context)
    
    # Create a simple chain
    rag_chain = (
        {"context": retriever, "question": lambda x: x["question"]}
        | qa_prompt
        | simple_answer_function
    )
    
    return rag_chain

def create_rag_chain(retriever: BaseRetriever):
    """Create a RAG QA chain using the retriever and QA capabilities"""
    lora_chain = create_lora_qa_chain(retriever)
    
    llm = setup_llm_model()
    
    if llm is None:
        return lora_chain
    
    # Define prompt template for QA
    template = """<|im_start|>system
You are a helpful AI assistant that answers questions based only on the provided context. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
<|im_end|>
<|im_start|>user
Use the following pieces of context to answer the question at the end.

Context:
{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
    
    qa_prompt = PromptTemplate.from_template(template)
    
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": qa_prompt
        },
        return_source_documents=True
    )
    
    class HybridRAGChain:
        def __init__(self, lora_chain, llm_chain):
            self.lora_chain = lora_chain
            self.llm_chain = llm_chain
            self.use_llm = True
        
        def invoke(self, inputs):
            # First, get the relevant documents
            if "query" in inputs:
                query = inputs["query"]
                question = inputs["query"]
            else:
                query = inputs["question"]
                question = inputs["question"]
            
            try:
                if hasattr(self.lora_chain, "invoke"):
                    lora_result = self.lora_chain.invoke({"question": question})
                    if isinstance(lora_result, dict):
                        lora_answer = lora_result.get("result", str(lora_result))
                    else:
                        lora_answer = str(lora_result)
                else:
                    try:
                        input_dict = {"question": question}
                        context_result = self.lora_chain["context"](input_dict)
                        function_result = self.lora_chain["simple_answer_function"]({
                            "question": question, 
                            "context": context_result
                        })
                        if isinstance(function_result, dict):
                            lora_answer = function_result.get("result", str(function_result))
                        else:
                            lora_answer = str(function_result)
                    except Exception as e:
                        print(f"Error with LoRA chain: {e}")
                        retriever = self.lora_chain["context"].retriever
                        documents = retriever.invoke(question)
                        lora_answer = extract_answer_from_docs(question, documents)
                
                if self.use_llm and hasattr(self.llm_chain, "invoke"):
                    try:
                        llm_result = self.llm_chain.invoke({"query": query})
                        source_docs = llm_result.get("source_documents", [])
                        
                        return {
                            "result": lora_answer,  
                            "llm_result": llm_result.get("result", ""),  
                            "source_documents": source_docs
                        }
                    except Exception as e:
                        print(f"Error using LLM chain: {e}")
                
                return {"result": lora_answer, "source_documents": []}
            
            except Exception as e:
                print(f"Error in hybrid chain: {e}")
                if self.use_llm and hasattr(self.llm_chain, "invoke"):
                    try:
                        return self.llm_chain.invoke({"query": query})
                    except Exception as e2:
                        print(f"Error falling back to LLM chain: {e2}")
                
                return {"result": "Unable to generate an answer due to technical issues."}
    
    return HybridRAGChain(lora_chain, llm_chain)

def main():
    """Main function to execute the ChromaDB and LangChain integration"""
    print("Loading datasets...")
    cv_data, qa_data = load_datasets()
    
    print("Preparing documents...")
    cv_documents = prepare_documents_from_cv(cv_data)
    qa_documents = prepare_qa_pairs(qa_data)
    all_documents = cv_documents + qa_documents
    
    print("Setting up ChromaDB...")
    client, collection = setup_chroma_client()
    
    if collection.count() == 0:
        print("Adding documents to collection...")
        add_documents_to_collection(collection, all_documents)
    else:
        print(f"Collection already contains {collection.count()} documents. Skipping document addition.")
    
    print("Setting up LangChain retriever and RAG chain...")
    retriever = setup_langchain_retriever(collection)
    rag_chain = create_rag_chain(retriever)
    
    example_queries = [
        "What was Andrew Cabrera's role at Dow Jones?",
        "What technologies does Andrew Cabrera know in Web Development?",
        "When did Andrew Cabrera attend Flatiron School?",
        "What are Andrew's responsibilities as a Freelance Software Engineer?"
    ]
    
    print("\n=== Running Example Queries ===")
    for query in example_queries:
        print(f"\nQuery: {query}")
        
        try:
            retrieved_docs = retriever.invoke(query)
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            if hasattr(rag_chain, "invoke") and callable(getattr(rag_chain, "invoke")):
                try:
                    result = rag_chain.invoke({"query": query})
                    print(f"Answer: {result['result']}")
                    if 'source_documents' in result:
                        print(f"Source Documents: {len(result['source_documents'])}")
                except KeyError:
                    result = rag_chain.invoke({"question": query})
                    print(f"Answer: {result}")
            else:
                result = rag_chain({"question": query})
                print(f"Answer: {result}")
        except Exception as e:
            print(f"Error processing query: {e}")
    
    print("\nRAG system setup complete!")

if __name__ == "__main__":
    main()
