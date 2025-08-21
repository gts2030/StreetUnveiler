#!/usr/bin/env python3
"""
Test script to verify uv environment installation for StreetUnveiler
"""

def test_pytorch():
    """Test PyTorch installation and CUDA availability"""
    print("🔧 Testing PyTorch...")
    import torch
    print(f"  ✅ PyTorch version: {torch.__version__}")
    print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA version: {torch.version.cuda}")
        print(f"  ✅ GPU count: {torch.cuda.device_count()}")
    print()

def test_submodules():
    """Test critical submodules"""
    print("📦 Testing submodules...")
    
    try:
        import tinycudann
        print("  ✅ Tiny-CUDA-NN")
    except ImportError as e:
        print(f"  ❌ Tiny-CUDA-NN: {e}")
    
    try:
        import diff_surfel_rasterization
        print("  ✅ Diff-Surfel-Rasterization")
    except ImportError as e:
        print(f"  ❌ Diff-Surfel-Rasterization: {e}")
    
    try:
        import superpose3d
        print("  ✅ Superpose3D")
    except ImportError as e:
        print(f"  ❌ Superpose3D: {e}")
    
    try:
        import sh_encoder
        print("  ✅ SH Encoder")
    except ImportError as e:
        print(f"  ❌ SH Encoder: {e}")
    
    try:
        import simple_knn
        from simple_knn._C import dist3knn, dist10knn, meanDistFromReferencePcd
        print("  ✅ Simple-KNN (with dist3knn, dist10knn, meanDistFromReferencePcd)")
    except ImportError as e:
        print(f"  ❌ Simple-KNN: {e}")
    
    try:
        import pandaset
        print("  ✅ PandaSet")
    except ImportError as e:
        print(f"  ❌ PandaSet: {e}")
    
    print()

def test_main_modules():
    """Test main StreetUnveiler modules"""
    print("🏠 Testing StreetUnveiler modules...")
    
    try:
        import arguments
        print("  ✅ Arguments")
    except ImportError as e:
        print(f"  ❌ Arguments: {e}")
    
    try:
        import gaussian_renderer
        print("  ✅ Gaussian Renderer")
    except ImportError as e:
        print(f"  ❌ Gaussian Renderer: {e}")
    
    try:
        import utils
        print("  ✅ Utils")
    except ImportError as e:
        print(f"  ❌ Utils: {e}")
    
    try:
        import scene
        print("  ✅ Scene")
    except ImportError as e:
        print(f"  ❌ Scene: {e}")
    
    print()

def test_training_readiness():
    """Test if environment is ready for training"""
    print("🚀 Testing training readiness...")
    
    try:
        # Test basic training imports without actually running training
        from arguments import ModelParams, PipelineParams, OptimizationParams
        print("  ✅ Training arguments can be imported")
    except ImportError as e:
        print(f"  ❌ Training arguments: {e}")
    
    try:
        # Test inpainting pipeline imports
        from simple_knn._C import meanDistFromReferencePcd
        print("  ✅ Inpainting pipeline dependencies available")
    except ImportError as e:
        print(f"  ❌ Inpainting pipeline: {e}")
    
    print()

if __name__ == "__main__":
    print("🧪 StreetUnveiler UV Environment Test")
    print("=" * 50)
    
    test_pytorch()
    test_submodules()
    test_main_modules()
    test_training_readiness()
    
    print("🎉 Environment test completed!")
    print("\n📋 To activate this environment in the future:")
    print("   source .venv/bin/activate")
    print("\n⚠️  Important: Always use LOCAL simple-knn submodule")
    print("   The local version contains required functions that are")
    print("   not available in the GitHub camenduru/simple-knn version")
