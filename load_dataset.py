from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

model = AutoModel.from_pretrained("NovaSearch/stella_en_400M_v5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("NovaSearch/stella_en_400M_v5", trust_remote_code=True)
dataset = load_dataset("json", data_files="dataset/cv_dataset.json", split="train", streaming=True)

def tokenize_dataset(dataset):
    return tokenizer(dataset, padding=True, truncation=True, return_tensors="pt", max_length=512)

dataset = dataset.map(tokenize_dataset, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()