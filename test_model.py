"""
Quick test script to verify the ML model implementation.
Tests ground truth loading, model training, and basic classification.
"""

from src.data.gee_fetcher import initialize_gee
from src.data.ground_truth_loader import GroundTruthLoader
from src.ml.model import CropStressModel
import json

def test_ground_truth_loader():
    """Test ground truth data loading."""
    print("=" * 60)
    print("TEST 1: Ground Truth Data Loader")
    print("=" * 60)
    
    try:
        loader = GroundTruthLoader('data/sample_ground_truth.csv')
        data = loader.load()
        
        print(f"✓ Loaded {len(data)} samples")
        print(f"✓ Stress classes: {data['stress_class'].unique()}")
        print(f"✓ Crop types: {data['crop_type'].unique()}")
        print(f"✓ Date range: {data['date'].min()} to {data['date'].max()}")
        
        # Test filtering
        loader.filter_by_crop_type(['wheat'])
        print(f"✓ Filtered to {len(loader.data)} wheat samples")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_training():
    """Test model training with ground truth data."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Training with Ground Truth")
    print("=" * 60)
    
    try:
        model = CropStressModel(use_spatial_features=False)
        model.train_from_ground_truth('data/sample_ground_truth.csv', num_trees=30)
        
        print("✓ Model trained successfully")
        print(f"✓ Training source: {model.training_data_source}")
        print(f"✓ Number of trees: {model.num_trees}")
        
        # Validate
        metrics = model.validate()
        print(f"✓ Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"✓ Kappa Coefficient: {metrics['kappa']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_training():
    """Test model training with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Training with Synthetic Data")
    print("=" * 60)
    
    try:
        model = CropStressModel(use_spatial_features=False)
        model.train_synthetic(num_trees=10)
        
        print("✓ Synthetic model trained successfully")
        
        # Validate
        metrics = model.validate()
        print(f"✓ Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ML MODEL ARCHITECTURE VERIFICATION TESTS")
    print("=" * 60 + "\n")
    
    # Initialize GEE
    try:
        initialize_gee()
        print("✓ GEE initialized successfully\n")
    except Exception as e:
        print(f"✗ GEE initialization failed: {e}")
        print("Please run: earthengine authenticate")
        return
    
    # Run tests
    results = {
        'ground_truth_loader': test_ground_truth_loader(),
        'model_training': test_model_training(),
        'synthetic_training': test_synthetic_training()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)


if __name__ == "__main__":
    main()
