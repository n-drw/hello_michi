#!/usr/bin/env python3
"""Compare first MatMulNBits output between ONNX Runtime and manual dequant."""

import onnx
import onnxruntime as ort
from onnx import numpy_helper
import numpy as np

# Load model
model_path = '/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b-qoperator/model.onnx'
model = onnx.load(model_path)

# Find first MatMulNBits and get its inputs
for node in model.graph.node:
    if node.op_type == 'MatMulNBits':
        print(f"Node: {node.name}")
        print(f"Inputs: {node.input}")
        
        # Get B and scales
        b_name = node.input[1]
        scales_name = node.input[2]
        
        for init in model.graph.initializer:
            if init.name == b_name:
                b_packed = numpy_helper.to_array(init).astype(np.float32)
                print(f"B shape: {b_packed.shape}")
            if init.name == scales_name:
                scales = numpy_helper.to_array(init)
                print(f"Scales shape: {scales.shape}")
        
        # Manual dequantization (like Burn does)
        # Unpack nibbles
        b_floor = np.floor(b_packed / 16.0)
        b_high = b_floor
        b_low = b_packed - b_floor * 16.0
        
        # Interleave: stack then reshape
        b_stacked = np.stack([b_low, b_high], axis=-1)  # [N, n_blocks, blob_size, 2]
        b_unpacked = b_stacked.reshape(b_packed.shape[0], b_packed.shape[1], -1)  # [N, n_blocks, block_size]
        print(f"Unpacked shape: {b_unpacked.shape}")
        
        # Center
        b_centered = b_unpacked - 8.0
        
        # Flatten to [N, K]
        b_flat = b_centered.reshape(b_packed.shape[0], -1)
        print(f"B flat shape: {b_flat.shape}")
        
        # Get K, N, block_size
        K = b_flat.shape[1]
        N = b_flat.shape[0]
        block_size = 128
        n_blocks = K // block_size
        print(f"K={K}, N={N}, n_blocks={n_blocks}")
        
        # Expand scales
        scales_2d = scales.reshape(N, n_blocks)
        scales_expanded = np.repeat(scales_2d, block_size, axis=1)  # [N, K]
        print(f"Scales expanded shape: {scales_expanded.shape}")
        
        # Dequantize
        b_dequant = b_flat * scales_expanded
        print(f"B dequant shape: {b_dequant.shape}")
        
        # Transpose for matmul: [K, N]
        b_weight = b_dequant.T
        print(f"B weight shape (for matmul): {b_weight.shape}")
        
        # Test with a simple input
        A = np.random.randn(1, 4, K).astype(np.float32)  # [batch, seq, K]
        
        # Manual matmul
        manual_out = A @ b_weight  # [1, 4, N]
        print(f"Manual output shape: {manual_out.shape}")
        print(f"Manual output sample: {manual_out[0, 0, :5]}")
        
        # Compare with ONNX Runtime on the full model (not directly comparable, but useful)
        print("\n--- Sanity check ---")
        print(f"B_dequant first row sample: {b_dequant[0, :8]}")
        print(f"B_dequant stats: min={b_dequant.min():.4f}, max={b_dequant.max():.4f}, mean={b_dequant.mean():.4f}")
        
        break

print("\n=== Done ===")

# Also check what tokens 238, 198, 15646 decode to
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b-qoperator/tokenizer.json')
print("\n--- Token decoding ---")
for tid in [238, 198, 15646]:
    decoded = tokenizer.decode([tid])
    print(f"Token {tid}: '{decoded}'")
