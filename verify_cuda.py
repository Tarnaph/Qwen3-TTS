
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    try:
        x = torch.rand(5, 3).cuda()
        print("Successfully created tensor on GPU.")
        print(x)
    except Exception as e:
        print(f"Error creating tensor on GPU: {e}")
else:
    print("CUDA is NOT available.")
