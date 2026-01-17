#!/usr/bin/env python3
"""
ONNX Opset Upgrade with External Data Support

Based on Burn's onnx_opset_upgrade.py but with proper handling for
models that use external data files (>2GB models).
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import onnx
    from onnx import shape_inference, version_converter
    from onnx.external_data_helper import convert_model_to_external_data
except ImportError:
    print("Error: onnx package not installed. Run: pip install onnx")
    sys.exit(1)


def load_onnx_model(model_path: Path, load_external: bool = True):
    """Load ONNX model with optional external data support."""
    try:
        model = onnx.load(str(model_path), load_external_data=load_external)
        print(f"Model loaded: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")


def get_opset_version(model):
    """Get the opset version of the model."""
    try:
        return model.opset_import[0].version
    except (IndexError, AttributeError):
        return None


def upgrade_opset(model, target_opset: int = 16):
    """Upgrade model to target opset version."""
    current_opset = get_opset_version(model)
    
    if current_opset is None:
        print("Warning: Could not determine current opset version")
        return model
    
    print(f"Current opset: {current_opset}")
    
    if current_opset >= target_opset:
        print(f"Opset {current_opset} already >= {target_opset}, skipping upgrade")
        return model
    
    try:
        upgraded = version_converter.convert_version(model, target_opset)
        print(f"Upgraded to opset {target_opset}")
        return upgraded
    except Exception as e:
        print(f"Warning: Opset upgrade failed: {e}")
        return model


def apply_shape_inference(model):
    """Apply ONNX shape inference to the model."""
    try:
        inferred = shape_inference.infer_shapes(model, data_prop=True)
        print("Shape inference applied successfully")
        return inferred
    except Exception as e:
        print(f"Warning: Shape inference partially applied: {e}")
        return model


def save_model_with_external_data(model, output_path: Path, external_data_name: str):
    """
    Save model that was loaded with external data.
    
    After shape inference, the model still has tensor data in memory.
    We need to write both the graph (to .onnx) and tensor data (to .onnx_data).
    """
    import tempfile
    import shutil
    
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use onnx.save_model with save_as_external_data=True
    # This properly handles models that have tensor data in memory
    try:
        onnx.save_model(
            model,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=1024,
        )
        
        # Verify the files were created correctly
        if output_path.stat().st_size < 100:
            raise RuntimeError(f"Model file too small: {output_path.stat().st_size} bytes")
        
        print(f"Model saved: {output_path}")
        print(f"External data: {output_path.parent / external_data_name}")
        
    except Exception as e:
        print(f"Error saving with external data: {e}")
        print("Attempting fallback save...")
        
        # Fallback: save to temp location then copy
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / output_path.name
            tmp_data_path = Path(tmpdir) / external_data_name
            
            onnx.save_model(
                model,
                str(tmp_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_name,
                size_threshold=1024,
            )
            
            if tmp_path.stat().st_size >= 100:
                shutil.copy2(str(tmp_path), str(output_path))
                if tmp_data_path.exists():
                    shutil.copy2(str(tmp_data_path), str(output_path.parent / external_data_name))
                print(f"Model saved (fallback): {output_path}")
            else:
                raise RuntimeError("Failed to save model with external data")


def save_model_inline(model, output_path: Path):
    """Save model with all data inline (for smaller models)."""
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx.save(model, str(output_path))
    print(f"Model saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Upgrade ONNX model opset and apply shape inference with external data support"
    )
    parser.add_argument("input", type=str, help="Input ONNX model path")
    parser.add_argument("-o", "--output", type=str, help="Output ONNX model path (default: input_opset16.onnx)")
    parser.add_argument("--opset", type=int, default=16, help="Target opset version (default: 16)")
    parser.add_argument("--inplace", action="store_true", help="Modify model in-place")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Check for external data
    external_data_path = input_path.with_suffix('.onnx_data')
    has_external_data = external_data_path.exists()
    
    if has_external_data:
        print(f"Detected external data file: {external_data_path}")
    
    # Determine output path
    if args.inplace:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            input_path.stem + f"_opset{args.opset}.onnx"
        )
    
    # Load model
    model = load_onnx_model(input_path, load_external=has_external_data)
    
    # Process model
    model = upgrade_opset(model, args.opset)
    model = apply_shape_inference(model)
    
    # Save model
    if has_external_data:
        external_name = output_path.with_suffix('.onnx_data').name
        save_model_with_external_data(model, output_path, external_name)
    else:
        save_model_inline(model, output_path)
    
    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()
