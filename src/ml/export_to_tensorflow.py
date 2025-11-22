"""
Export PyTorch model to TensorFlow for GEE deployment.
Converts PyTorch → ONNX → TensorFlow → SavedModel format.

Author: Nganga Kamau
GitHub: https://github.com/NgangaKamau3/AgriSense
"""

import torch
import torch.onnx
import onnx
import tensorflow as tf
from pathlib import Path
import numpy as np
from typing import Tuple

from src.ml.efficient_stress_net import create_model


def export_to_onnx(pytorch_model: torch.nn.Module,
                   output_path: str,
                   input_shape: Tuple[int, int, int, int] = (1, 5, 256, 256)):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        pytorch_model: Trained PyTorch model
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    print(f"Exporting PyTorch model to ONNX...")
    
    # Set model to evaluation mode
    pytorch_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,  # GEE compatible
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"✓ ONNX model saved to {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")


def onnx_to_tensorflow(onnx_path: str, tf_output_dir: str):
    """
    Convert ONNX model to TensorFlow SavedModel.
    
    Args:
        onnx_path: Path to ONNX model
        tf_output_dir: Directory to save TensorFlow model
    """
    print(f"Converting ONNX to TensorFlow...")
    
    try:
        import onnx_tf
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Export as SavedModel
        tf_rep.export_graph(tf_output_dir)
        
        print(f"✓ TensorFlow model saved to {tf_output_dir}")
        
    except ImportError:
        print("Error: onnx-tf not installed")
        print("Install with: pip install onnx-tf")
        raise


def optimize_for_gee(tf_model_dir: str, optimized_output_dir: str):
    """
    Optimize TensorFlow model for GEE deployment.
    Applies quantization and pruning to reduce model size.
    
    Args:
        tf_model_dir: Path to TensorFlow SavedModel
        optimized_output_dir: Path to save optimized model
    """
    print("Optimizing model for GEE deployment...")
    
    # Load model
    model = tf.saved_model.load(tf_model_dir)
    
    # Get concrete function
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    # Convert to TFLite with optimizations
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Use FP16 for smaller size
    
    # Convert
    tflite_model = converter.convert()
    
    # Save optimized model
    optimized_path = Path(optimized_output_dir) / 'optimized_model.tflite'
    optimized_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(optimized_path, 'wb') as f:
        f.write(tflite_model)
    
    # Calculate size reduction
    import os
    original_size = sum(
        os.path.getsize(os.path.join(tf_model_dir, f))
        for f in os.listdir(tf_model_dir)
        if os.path.isfile(os.path.join(tf_model_dir, f))
    ) / 1024 / 1024  # MB
    
    optimized_size = os.path.getsize(optimized_path) / 1024 / 1024  # MB
    
    print(f"✓ Optimized model saved to {optimized_path}")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Optimized size: {optimized_size:.2f} MB")
    print(f"  Size reduction: {(1 - optimized_size/original_size)*100:.1f}%")


def export_pytorch_to_gee(pytorch_model_path: str,
                         output_dir: str = 'models/gee_export',
                         model_size: str = 'medium'):
    """
    Complete export pipeline: PyTorch → ONNX → TensorFlow → Optimized.
    
    Args:
        pytorch_model_path: Path to PyTorch checkpoint (.pth)
        output_dir: Directory to save exported models
        model_size: Model size ('small', 'medium', 'large')
    
    Returns:
        Dictionary with paths to exported models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PyTorch to GEE Export Pipeline")
    print("=" * 60)
    
    # 1. Load PyTorch model
    print("\n1. Loading PyTorch model...")
    model = create_model(model_size=model_size, num_classes=5)
    
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ PyTorch model loaded")
    
    # 2. Export to ONNX
    print("\n2. Exporting to ONNX...")
    onnx_path = output_dir / 'model.onnx'
    export_to_onnx(model, str(onnx_path))
    
    # 3. Convert to TensorFlow
    print("\n3. Converting to TensorFlow...")
    tf_dir = output_dir / 'tensorflow_model'
    onnx_to_tensorflow(str(onnx_path), str(tf_dir))
    
    # 4. Optimize for GEE
    print("\n4. Optimizing for GEE...")
    optimized_dir = output_dir / 'optimized'
    optimize_for_gee(str(tf_dir), str(optimized_dir))
    
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    
    export_info = {
        'pytorch_model': str(pytorch_model_path),
        'onnx_model': str(onnx_path),
        'tensorflow_model': str(tf_dir),
        'optimized_model': str(optimized_dir / 'optimized_model.tflite')
    }
    
    # Save export info
    import json
    with open(output_dir / 'export_info.json', 'w') as f:
        json.dump(export_info, f, indent=2)
    
    print("\nNext steps for GEE deployment:")
    print("1. Upload TensorFlow model to Google AI Platform:")
    print(f"   gcloud ai-platform models create stress_classifier")
    print(f"   gcloud ai-platform versions create v1 \\")
    print(f"     --model stress_classifier \\")
    print(f"     --origin={tf_dir} \\")
    print(f"     --runtime-version=2.11 \\")
    print(f"     --python-version=3.9")
    print("\n2. Use the model in GEE with ee.Model.fromAiPlatformPredictor()")
    
    return export_info


def test_exported_model(onnx_path: str):
    """
    Test ONNX model to verify export correctness.
    
    Args:
        onnx_path: Path to ONNX model
    """
    print("\nTesting exported ONNX model...")
    
    import onnxruntime as ort
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = np.random.randn(1, 5, 256, 256).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {'input': test_input})
    
    print(f"✓ ONNX model test successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        pytorch_model_path = sys.argv[1]
    else:
        pytorch_model_path = 'models/pytorch/best_model.pth'
    
    print(f"Exporting model from: {pytorch_model_path}")
    
    # Run export pipeline
    export_info = export_pytorch_to_gee(
        pytorch_model_path=pytorch_model_path,
        output_dir='models/gee_export',
        model_size='medium'
    )
    
    # Test exported model
    test_exported_model(export_info['onnx_model'])
    
    print("\n✓ Export pipeline completed successfully!")
