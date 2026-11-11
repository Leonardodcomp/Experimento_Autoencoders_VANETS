import sys, torch, sklearn, scipy

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA do PyTorch:", torch.version.cuda)
print("GPU disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("Teste na GPU OK:", (x @ x).shape)

print("scikit-learn:", sklearn.__version__)
print("SciPy:", scipy.__version__)