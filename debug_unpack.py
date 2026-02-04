#!/usr/bin/env python3
"""Debug script to verify 4-bit unpacking logic."""

import onnx
from onnx import numpy_helper, AttributeProto
import numpy as np

model = onnx.load('/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b-qoperator/model.onnx')

for node in model.graph.node:
    if node.op_type == 'MatMulNBits':
        # Check attribute types
        print("Raw attributes:")
        for attr in node.attribute:
            attr_type = AttributeProto.AttributeType.Name(attr.type)
            if attr.type == AttributeProto.INT:
                print(f"  {attr.name}: {attr.i} (type={attr_type})")
            elif attr.type == AttributeProto.FLOAT:
                print(f"  {attr.name}: {attr.f} (type={attr_type})")
            else:
                print(f"  {attr.name}: type={attr_type}")
        
        attrs = {a.name: a.i for a in node.attribute}
        print(f"\nParsed: K={attrs.get('K')}, N={attrs.get('N')}, block_size={attrs.get('block_size')}")
        
        b_name = node.input[1]
        scales_name = node.input[2]
        
        for init in model.graph.initializer:
            if init.name == b_name:
                b = numpy_helper.to_array(init)
                print(f"B shape: {b.shape}, dtype: {b.dtype}")
                print(f"First 4 bytes: {b.flatten()[:4]}")
                
                # Unpack using float div/floor (like Burn does)
                bf = b.astype(np.float32)
                b_floor = np.floor(bf / 16.0)
                b_high = b_floor
                b_low = bf - b_floor * 16.0
                
                # Stack and reshape (like Burn's stack then flatten)
                stacked = np.stack([b_low, b_high], axis=-1)
                unpacked = stacked.reshape(b.shape[0], b.shape[1], -1)
                print(f"Unpacked shape: {unpacked.shape}")
                print(f"Unpacked first 8 values: {unpacked[0, 0, :8]}")
                
                # Bitwise reference (ground truth)
                print("\nBitwise unpacking (ground truth):")
                for i, bv in enumerate(b.flatten()[:4]):
                    low = bv & 0xF
                    high = bv >> 4
                    print(f"  byte {bv}: low={low}, high={high}")
                
                # Center by subtracting 8
                centered = unpacked - 8.0
                print(f"\nCentered first 8 values: {centered[0, 0, :8]}")
                break
        
        for init in model.graph.initializer:
            if init.name == scales_name:
                scales = numpy_helper.to_array(init)
                print(f"\nScales shape: {scales.shape}, dtype: {scales.dtype}")
                print(f"Scales first 8: {scales[:8]}")
                break
        
        break

print("\n=== Done ===")
