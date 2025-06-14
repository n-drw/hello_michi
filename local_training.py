from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch
from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset/qa_dataset.json")
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

def preprocess_function(examples):
    questions = examples.get("question", examples.get("questions", []))
    contexts = examples.get("context", examples.get("contexts", examples.get("answer", examples.get("answers", []))))
    answers = examples.get("answer", examples.get("answers", []))
    
    tokenized_inputs = tokenizer(
        questions,
        contexts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    start_positions = []
    end_positions = []

    for i, (question, context, answer) in enumerate(zip(questions, contexts, answers)):
        answer_start = context.find(answer)
        if answer_start == -1:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_token = tokenized_inputs.char_to_token(i, answer_start)
            end_token = tokenized_inputs.char_to_token(i, answer_start + len(answer) - 1)

            if start_token is None:
                start_token = 0
            if end_token is None:
                end_token = 0
            
            start_positions.append(start_token)
            end_positions.append(end_token)

    tokenized_inputs["start_positions"] = start_positions
    tokenized_inputs["end_positions"] = end_positions

    return tokenized_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names 
)

model = AutoModelForQuestionAnswering.from_pretrained('intfloat/multilingual-e5-large')
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.QUESTION_ANS
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora-results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    max_steps=1000,
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=None,  
)

trainer.train()
trainer.save_model("./lora-sentence-transformer")