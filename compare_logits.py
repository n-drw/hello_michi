#!/usr/bin/env python3
"""Compare logits between ONNX Runtime and expected values."""

import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file('/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b-qoperator/tokenizer.json')

# Tokenize the same prompt as Burn
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
encoding = tokenizer.encode(prompt)
input_ids = np.array([encoding.ids], dtype=np.int64)
print(f"Input IDs shape: {input_ids.shape}")
print(f"First 4 input IDs: {input_ids[0, :4]}")

seq_len = input_ids.shape[1]
batch_size = 1

# Create other inputs - NO dummy cache token
attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)  # Just seq_len, no dummy
position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

# Create empty KV cache with seq_len=0 (no dummy token)
num_layers = 28
num_kv_heads = 8
head_dim = 128
past_key_values = {}
for i in range(num_layers):
    # Use seq_len=0 for initial empty cache
    past_key_values[f'past_key_values.{i}.key'] = np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=np.float32)
    past_key_values[f'past_key_values.{i}.value'] = np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=np.float32)

# Run ONNX model
model_path = '/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b-qoperator/model.onnx'
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Get input names
input_names = [inp.name for inp in session.get_inputs()]
print(f"\nModel inputs: {input_names[:5]}...")

# Prepare feeds
feeds = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'position_ids': position_ids,
}
feeds.update(past_key_values)

# Run inference
outputs = session.run(None, feeds)
logits = outputs[0]

print(f"\nLogits shape: {logits.shape}")
print(f"Last token logits (first 10): {logits[0, -1, :10]}")
print(f"Logits range: min={logits.min():.4f}, max={logits.max():.4f}")

# Get argmax token
next_token = np.argmax(logits[0, -1, :])
print(f"\nArgmax token ID: {next_token}")
print(f"Decoded: '{tokenizer.decode([next_token])}'")

# Compare with Burn's output
print("\n--- Comparison with Burn ---")
print("Burn first 10 logits: [-2.68, -0.19, -3.12, -2.28, -5.13, 1.92, 4.29, 3.02, -2.62, 5.07]")
print("Burn logits range: min=-20.02, max=15.29")
print("Burn argmax token: 238")
