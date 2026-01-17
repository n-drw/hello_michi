#!/usr/bin/env python3
"""
Decomposed ONNX Export for Burn Compatibility

This script exports transformer models to ONNX format with decomposed operators,
making them compatible with Burn's ONNX importer which doesn't support:
- RMSNormalization
- RotaryEmbedding  
- Swish

By using decomposed ops, these are broken down into primitives that Burn supports:
- RMSNorm -> Mul, ReduceMean, Sqrt, Div
- RoPE -> Sin, Cos, Mul, Concat, Slice
- Swish/SiLU -> Sigmoid, Mul

Usage:
    python export_onnx_decomposed.py --model Qwen/Qwen3-0.6B --output ./onnx_models/qwen3-0.6b
    python export_onnx_decomposed.py --model gpt2 --output ./onnx_models/gpt2
"""

import argparse
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


ALTERNATIVE_MODELS = {
    "gpt2": {
        "name": "GPT-2 (124M)",
        "hf_id": "gpt2",
        "params": "124M",
        "burn_compatible": "high",
        "notes": "Uses LayerNorm, learned pos embeddings. Fully supported by Burn ONNX."
    },
    "gpt2-medium": {
        "name": "GPT-2 Medium (355M)",
        "hf_id": "gpt2-medium",
        "params": "355M",
        "burn_compatible": "high",
        "notes": "Larger GPT-2 variant, same architecture."
    },
    "distilgpt2": {
        "name": "DistilGPT-2 (82M)",
        "hf_id": "distilgpt2",
        "params": "82M",
        "burn_compatible": "high",
        "notes": "Distilled GPT-2, smaller and faster."
    },
    
    "phi-1_5": {
        "name": "Phi-1.5 (1.3B)",
        "hf_id": "microsoft/phi-1_5",
        "params": "1.3B",
        "burn_compatible": "medium",
        "notes": "Uses RoPE but may export with decomposed ops. Very capable for size."
    },
    
    "opt-125m": {
        "name": "OPT-125M",
        "hf_id": "facebook/opt-125m",
        "params": "125M",
        "burn_compatible": "high",
        "notes": "Uses LayerNorm, learned pos embeddings. Good Burn compatibility."
    },
    "opt-350m": {
        "name": "OPT-350M",
        "hf_id": "facebook/opt-350m",
        "params": "350M",
        "burn_compatible": "high",
        "notes": "Larger OPT variant."
    },
    
    "bloom-560m": {
        "name": "BLOOM-560M",
        "hf_id": "bigscience/bloom-560m",
        "params": "560M",
        "burn_compatible": "medium",
        "notes": "Uses ALiBi positional encoding, may need verification."
    },
    
    "qwen3-0.6b": {
        "name": "Qwen3-0.6B",
        "hf_id": "Qwen/Qwen3-0.6B",
        "params": "600M",
        "burn_compatible": "low-medium",
        "notes": "Requires decomposed export. Uses RMSNorm, RoPE, SwiGLU."
    },
    
    "tinyllama": {
        "name": "TinyLlama-1.1B",
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params": "1.1B",
        "burn_compatible": "low-medium",
        "notes": "LLaMA architecture, requires decomposed export."
    },
    
    "smollm-135m": {
        "name": "SmolLM-135M",
        "hf_id": "HuggingFaceTB/SmolLM-135M",
        "params": "135M",
        "burn_compatible": "medium",
        "notes": "Designed for edge deployment, check ONNX compatibility."
    },
}

def list_alternative_models():
    """Print a formatted list of alternative models."""
    print("\n" + "=" * 80)
    print("ALTERNATIVE MODELS FOR BURN COMPATIBILITY")
    print("=" * 80)
    
    high_compat = []
    medium_compat = []
    low_compat = []
    
    for key, model in ALTERNATIVE_MODELS.items():
        compat = model["burn_compatible"]
        if "high" in compat:
            high_compat.append((key, model))
        elif "medium" in compat:
            medium_compat.append((key, model))
        else:
            low_compat.append((key, model))
    
    print("\n🟢 HIGH COMPATIBILITY (Recommended for Burn):")
    print("-" * 60)
    for key, model in high_compat:
        print(f"  {model['name']:25} | {model['params']:8} | {model['hf_id']}")
        print(f"    └─ {model['notes']}")
    
    print("\n🟡 MEDIUM COMPATIBILITY (May require decomposed export):")
    print("-" * 60)
    for key, model in medium_compat:
        print(f"  {model['name']:25} | {model['params']:8} | {model['hf_id']}")
        print(f"    └─ {model['notes']}")
    
    print("\n🔴 LOW COMPATIBILITY (Requires decomposed export + verification):")
    print("-" * 60)
    for key, model in low_compat:
        print(f"  {model['name']:25} | {model['params']:8} | {model['hf_id']}")
        print(f"    └─ {model['notes']}")
    
    print("\n" + "=" * 80)


def run_shape_inference(onnx_path: str) -> bool:
    """
    Run ONNX shape inference on the model.
    
    This is critical for Burn ONNX import - without shape inference,
    intermediate tensor shapes are unknown and type inference fails.
    """
    try:
        import onnx
        from onnx import shape_inference
        from onnx.external_data_helper import convert_model_to_external_data
        from pathlib import Path
        import shutil
    except ImportError:
        print("⚠️  onnx package not installed. Run: pip install onnx")
        return False
    
    print(f"\n🔄 Running ONNX shape inference on {onnx_path}...")
    
    try:
        onnx_path = Path(onnx_path)
        model_dir = onnx_path.parent
        
        # Check if model uses external data
        external_data_path = onnx_path.with_suffix('.onnx_data')
        has_external_data = external_data_path.exists()
        
        # Load model with external data support
        model = onnx.load(str(onnx_path), load_external_data=True)
        
        # Run shape inference
        model = shape_inference.infer_shapes(model, data_prop=True)
        
        # Save back - handle external data properly
        if has_external_data:
            # For models with external data, we need to:
            # 1. Remove old files
            # 2. Convert model to use external data
            # 3. Save to the same location
            
            # Backup and remove old external data
            backup_data = model_dir / "model_backup.onnx_data"
            if external_data_path.exists():
                shutil.move(str(external_data_path), str(backup_data))
            
            try:
                # Convert tensors to external data format
                convert_model_to_external_data(
                    model,
                    all_tensors_to_one_file=True,
                    location=external_data_path.name,
                    size_threshold=1024,
                )
                
                # Save the model (this creates new external data file)
                onnx.save(model, str(onnx_path))
                
                # Remove backup on success
                if backup_data.exists():
                    backup_data.unlink()
                    
            except Exception as e:
                # Restore backup on failure
                if backup_data.exists():
                    shutil.move(str(backup_data), str(external_data_path))
                raise e
        else:
            onnx.save(model, str(onnx_path))
        
        print(f"✅ Shape inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️  Shape inference failed: {e}")
        print("   This may cause issues with Burn ONNX import.")
        import traceback
        traceback.print_exc()
        return False


def check_onnx_operators(onnx_path: str) -> Dict[str, List[str]]:
    """Analyze ONNX model for unsupported operators."""
    try:
        import onnx
    except ImportError:
        print("⚠️  onnx package not installed. Run: pip install onnx")
        return {}
    
    model = onnx.load(onnx_path, load_external_data=False)
    
    # Operators that burn-import does NOT support
    unsupported_ops = {
        "RMSNormalization", "RotaryEmbedding", "Swish", "Mish", "Elu",
        "Softplus", "Softsign", "Selu", "Celu", "DequantizeLinear",
        "QuantizeLinear", "QLinearConv", "QLinearMatMul", "Einsum",
        "ScatterElements", "ScatterND", "GatherND"
    }
    
    operators_used = set()
    for node in model.graph.node:
        operators_used.add(node.op_type)
    
    found_unsupported = operators_used.intersection(unsupported_ops)
    found_supported = operators_used - unsupported_ops
    
    return {
        "supported": sorted(list(found_supported)),
        "unsupported": sorted(list(found_unsupported)),
        "all": sorted(list(operators_used))
    }


def export_with_optimum(
    model_id: str,
    output_dir: str,
    opset_version: int = 17,
    no_post_process: bool = True,
    fp16: bool = False,
) -> bool:
    """
    Export model using HuggingFace Optimum with decomposed operators.
    
    This is the recommended approach as Optimum handles the complexity
    of exporting transformer models properly.
    """
    try:
        from optimum.exporters.onnx import main_export
        from optimum.exporters.onnx.model_configs import TextDecoderOnnxConfig
    except ImportError:
        print("⚠️  optimum package not installed.")
        print("   Run: pip install optimum[exporters]")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Exporting {model_id} with Optimum...")
    print(f"   Output: {output_path}")
    print(f"   Opset: {opset_version}")
    print(f"   No post-process (keeps decomposed ops): {no_post_process}")
    print(f"   FP16 (half precision): {fp16}")
    
    # Use simpler task without KV cache for smaller model size
    task = "text-generation"  # No KV cache - smaller but slower inference
    
    try:
        main_export(
            model_name_or_path=model_id,
            output=output_path,
            task=task,
            opset=opset_version,
            no_post_process=no_post_process,  # Prevent operator fusion
            fp16=fp16,  # Half precision reduces size by 50%
        )
        
        print(f"✅ Export completed successfully!")
        
        # Check for unsupported operators
        onnx_file = output_path / "model.onnx"
        if onnx_file.exists():
            # Note: Shape inference should be run separately using Burn's onnx_opset_upgrade.py
            # For models with external data, the script needs special handling
            
            print("\n🔍 Analyzing exported model for Burn compatibility...")
            ops = check_onnx_operators(str(onnx_file))
            
            if ops.get("unsupported"):
                print(f"\n⚠️  Found unsupported operators: {ops['unsupported']}")
                print("   These may need manual implementation in Burn.")
            else:
                print(f"\n✅ All operators are supported by Burn ONNX import!")
            
            print(f"\n📊 Operators used: {len(ops.get('all', []))}")
            
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def export_with_torch(
    model_id: str,
    output_dir: str,
    opset_version: int = 17,
    sequence_length: int = 512,
) -> bool:
    """
    Export model using native PyTorch ONNX export.
    
    This gives more control but may not handle all transformer
    architectures as well as Optimum.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Loading model {model_id}...")
    
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.eval()
        
        print(f"✅ Model loaded: {config.model_type}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        dummy_input = tokenizer(
            "Hello, I am a test input for ONNX export.",
            return_tensors="pt",
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
        )
        
        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]
        
        print(f"\n🔄 Exporting to ONNX (opset {opset_version})...")
        
        onnx_path = output_path / "model.onnx"
        
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            # These options help with decomposition
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )
        
        print(f"✅ Export completed: {onnx_path}")
        
        # Run shape inference for Burn compatibility
        run_shape_inference(str(onnx_path))
        
        tokenizer.save_pretrained(output_path)
        print(f"✅ Tokenizer saved")
        
        print("\n🔍 Analyzing exported model for Burn compatibility...")
        ops = check_onnx_operators(str(onnx_path))
        
        if ops.get("unsupported"):
            print(f"\n⚠️  Found unsupported operators: {ops['unsupported']}")
        else:
            print(f"\n✅ All operators appear supported by Burn ONNX import!")
        
        # Save model info
        info = {
            "model_id": model_id,
            "model_type": config.model_type,
            "parameters": sum(p.numel() for p in model.parameters()),
            "opset_version": opset_version,
            "sequence_length": sequence_length,
            "operators": ops,
        }
        
        with open(output_path / "model_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export transformer models to ONNX with decomposed operators for Burn compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Export Qwen3-0.6B with decomposed operators
            python export_onnx_decomposed.py --model Qwen/Qwen3-0.6B --output ./onnx_models/qwen3

            # Export GPT-2 (high compatibility)
            python export_onnx_decomposed.py --model gpt2 --output ./onnx_models/gpt2

            # List alternative models
            python export_onnx_decomposed.py --list-alternatives

            # Check an existing ONNX model
            python export_onnx_decomposed.py --check ./onnx_models/model.onnx
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="HuggingFace model ID (e.g., 'gpt2', 'Qwen/Qwen3-0.6B')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./onnx_models",
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, Burn recommends 16+)"
    )
    parser.add_argument(
        "--method",
        choices=["optimum", "torch"],
        default="optimum",
        help="Export method: 'optimum' (recommended) or 'torch'"
    )
    parser.add_argument(
        "--list-alternatives",
        action="store_true",
        help="List alternative models with better Burn compatibility"
    )
    parser.add_argument(
        "--check",
        type=str,
        metavar="ONNX_PATH",
        help="Check an existing ONNX model for unsupported operators"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for export (torch method only)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export model in FP16 (half precision) - reduces size by 50%%"
    )
    
    args = parser.parse_args()
    
    if args.list_alternatives:
        list_alternative_models()
        return
    
    if args.check:
        print(f"\n🔍 Checking {args.check} for Burn compatibility...")
        ops = check_onnx_operators(args.check)
        
        print(f"\n📊 Total operators: {len(ops.get('all', []))}")
        print(f"\n✅ Supported operators ({len(ops.get('supported', []))}):")
        for op in ops.get("supported", []):
            print(f"   - {op}")
        
        if ops.get("unsupported"):
            print(f"\n❌ Unsupported operators ({len(ops['unsupported'])}):")
            for op in ops["unsupported"]:
                print(f"   - {op}")
        else:
            print("\n✅ All operators are supported by Burn!")
        return
    
    # Validate model argument
    if not args.model:
        parser.error("--model is required unless using --list-alternatives or --check")
    
    # Perform export
    print("=" * 60)
    print("ONNX EXPORT FOR BURN COMPATIBILITY")
    print("=" * 60)
    
    if args.method == "optimum":
        success = export_with_optimum(
            model_id=args.model,
            output_dir=args.output,
            opset_version=args.opset,
            fp16=args.fp16,
        )
    else:
        success = export_with_torch(
            model_id=args.model,
            output_dir=args.output,
            opset_version=args.opset,
            sequence_length=args.seq_length,
        )
    
    if success:
        print("\n" + "=" * 60)
        print("✅ EXPORT COMPLETE")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Check model_info.json for operator compatibility")
        print(f"  2. If unsupported ops found, try a high-compatibility model:")
        print(f"     python export_onnx_decomposed.py --list-alternatives")
        print(f"  3. Use the ONNX model with Burn:")
        print(f"     - Add to build.rs: ModelGen::new().input(\"{args.output}/model.onnx\")")
        print(f"     - See: https://burn.dev/books/burn/import/onnx-model.html")
    else:
        print("\n" + "=" * 60)
        print("❌ EXPORT FAILED")
        print("=" * 60)
        print("\nTry an alternative model with better compatibility:")
        print("  python export_onnx_decomposed.py --list-alternatives")


if __name__ == "__main__":
    main()
