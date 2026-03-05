"""
GPU Availability Check
This script checks if PyTorch can access a GPU (CUDA) for faster training.

What this means:
- GPU available (CUDA): Training will be FAST (minutes for 3 epochs)
- CPU only: Training will be SLOW (could take 1-2 hours for 3 epochs)
"""

import torch

print("=" * 60)
print("GPU AVAILABILITY CHECK")
print("=" * 60)

# Check if CUDA (NVIDIA GPU) is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("✅ GPU DETECTED!")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    print("\n📊 Training will be FAST (~5-20 minutes for 3 epochs)")
else:
    print("❌ NO GPU DETECTED")
    print("   Training will use CPU (slower)")
    print("\n⏱️  Training will be SLOWER (~30-120 minutes for 3 epochs)")
    print("   Consider using Google Colab (free GPU) for faster training:")
    print("   https://colab.research.google.com/")

print("\n" + "=" * 60)

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# Check if MPS (Mac) is available
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✅ Apple Silicon (MPS) detected - can use GPU acceleration")

print("=" * 60)
