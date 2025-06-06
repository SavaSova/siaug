import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("------------------")
print(torch.__version__)
print(torch.version.cuda)
print("------------------")
print("CUDA available:", torch.cuda.is_available())
print("Device in use:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
