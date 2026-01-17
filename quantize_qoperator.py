#!/usr/bin/env python3
"""
Quantize ONNX model to QOperator format using MatMulNBitsQuantizer.
This produces MatMulNBits nodes instead of DequantizeLinear (QDQ format).
"""

import os
from pathlib import Path

# MatMulNBits quantizer from onnxruntime
from onnxruntime.quantization import matmul_nbits_quantizer

def quantize_to_qoperator(input_model: str, output_dir: str, block_size: int = 128):
    """Quantize model to QOperator format with MatMulNBits."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_model = str(output_path / "model.onnx")
    
    print(f"Input model: {input_model}")
    print(f"Output model: {output_model}")
    print(f"Block size: {block_size}")
    print(f"Using MatMulNBitsQuantizer (QOperator format)")
    
    # Create quantizer
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model=input_model,
        block_size=block_size,
        is_symmetric=True,  # Symmetric quantization (Q4S)
        accuracy_level=4,   # Higher accuracy
    )
    
    print("Processing model...")
    quantizer.process()
    
    print(f"Saving quantized model to {output_model}")
    quantizer.model.save_model_to_file(output_model, use_external_data_format=True)
    
    # Check output size
    if os.path.exists(output_model):
        size_mb = os.path.getsize(output_model) / (1024 * 1024)
        print(f"Output model size: {size_mb:.2f} MB")
    
    # Check for external data file
    data_file = output_model + "_data"
    if os.path.exists(data_file):
        data_size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"External data size: {data_size_mb:.2f} MB")
        print(f"Total size: {size_mb + data_size_mb:.2f} MB")
    
    print("Done!")
    return output_model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize ONNX model to QOperator format")
    parser.add_argument("--input", "-i", 
                       default="/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b/model.onnx",
                       help="Input ONNX model path")
    parser.add_argument("--output", "-o",
                       default="/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b-qoperator",
                       help="Output directory for quantized model")
    parser.add_argument("--block-size", "-b", type=int, default=128,
                       help="Block size for quantization (default: 128)")
    
    args = parser.parse_args()
    
    quantize_to_qoperator(args.input, args.output, args.block_size)
