import torch

# Check if CUDA is available
print("CUDA Available: ", torch.cuda.is_available())

# Check if cuDNN is available
print("cuDNN Enabled: ", torch.backends.cudnn.enabled)

# Check PyTorch CUDA version and cuDNN version
print("CUDA Version: ", torch.version.cuda)
print("cuDNN Version: ", torch.backends.cudnn.version())
