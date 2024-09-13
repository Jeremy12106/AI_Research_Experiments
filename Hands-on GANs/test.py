

# CUDA測試工具
import torch
if torch.cuda.is_available():
    print("CUDA 可用！")
    print(f"PyTorch CUDA 版本: {torch.version.cuda}")
    print(f"設備數量: {torch.cuda.device_count()}")
    print(f"當前設備: {torch.cuda.current_device()}")
    print(f"設備名稱: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA 不可用，使用 CPU")

x = torch.tensor([1.0, 2.0, 3.0])
if torch.cuda.is_available():
    x = x.to('cuda')
    print("成功將 Tensor 移動到 CUDA")
else:
    print("CUDA 不可用，Tensor 保持在 CPU")