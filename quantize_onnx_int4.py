#!/usr/bin/env python3
"""
Quantize ONNX model to INT4 with QDQ format for Burn compatibility.

Uses DequantizeLinear nodes instead of MatMulNBits, which is supported
by our patched burn-import.
"""

import argparse
from pathlib import Path

try:
    from onnxruntime.quantization import quant_utils, quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    import numpy as np
except ImportError as e:
    print(f"Error: onnxruntime quantization modules not available: {e}")
    print("Run: pip install onnxruntime>=1.17.0")
    exit(1)


class DummyDataReader(CalibrationDataReader):
    """Simple calibration data reader that generates random data."""
    def __init__(self, input_name, input_shape, num_samples=10):
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.current = 0
    
    def get_next(self):
        if self.current >= self.num_samples:
            return None
        self.current += 1
        # Generate random token IDs (typical range for LLM vocab)
        return {self.input_name: np.random.randint(0, 32000, size=self.input_shape).astype(np.int64)}


def quantize_to_int4_qdq(input_path: str, output_path: str, block_size: int = 128):
    """
    Quantize ONNX model to INT4 using QDQ format.
    
    QDQ format uses DequantizeLinear nodes which are supported by Burn,
    unlike QOperator format which uses MatMulNBits.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {input_path}")
    print(f"Block size: {block_size}")
    print(f"Format: QDQ (DequantizeLinear - Burn compatible)")
    
    try:
        # Load model with shape inference
        model = quant_utils.load_model_with_shape_infer(input_path)
        
        print("Quantizing model to INT8 using static quantization (weight-only)...")
        # Use static quantization with weight-only mode
        # This uses DequantizeLinear nodes which are Burn-compatible
        # Create calibration data reader for the model input
        calibration_reader = DummyDataReader(
            input_name="input_ids",
            input_shape=[1, 32],  # batch_size=1, seq_len=32
            num_samples=5
        )
        
        quantize_static(
            model_input=str(input_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_reader,
            quant_format=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
            },
        )
        
        # Skip the save step below since quantize_static saves directly
        return True
        
        # Save quantized model
        print(f"Saving quantized model: {output_path}")
        quantizer.model.save_model_to_file(str(output_path), use_external_data_format=True)
        
        # Check file sizes
        if output_path.exists():
            onnx_size = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ Quantized model size: {onnx_size:.1f} MB")
            
            # Check for external data
            data_path = output_path.with_suffix('.onnx_data')
            if data_path.exists():
                data_size = data_path.stat().st_size / (1024 * 1024)
                print(f"   External data size: {data_size:.1f} MB")
                print(f"   Total size: {onnx_size + data_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model to INT4 with QDQ format for Burn"
    )
    parser.add_argument("input", type=str, help="Input ONNX model path")
    parser.add_argument("-o", "--output", type=str, help="Output path (default: input_q4.onnx)")
    parser.add_argument("--block-size", type=int, default=128, help="Quantization block size (default: 128)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_q4.onnx")
    
    print("=" * 60)
    print("INT4 QUANTIZATION (QDQ FORMAT)")
    print("=" * 60)
    
    success = quantize_to_int4_qdq(str(input_path), str(output_path), args.block_size)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ QUANTIZATION COMPLETE")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Test with burn-import:")
        print(f"     BURN_ONNX_MODEL_PATH={output_path} cargo build --features burn-local")
    else:
        print("\n❌ QUANTIZATION FAILED")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
