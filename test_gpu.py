import torch

print("--- DIAGNOSTICA GPU ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

print("\n1. Test Allocazione Base PyTorch...")
try:
    x = torch.zeros((1000, 1000)).cuda()
    print("✅ Allocazione Base SUCCESSO!")
except Exception as e:
    print("❌ Allocazione Base FALLITA:", e)

print("\n2. Test Modello Lineare Base...")
try:
    from torch.nn import Linear

    layer = Linear(100, 100).cuda()
    print("✅ Modello Base SUCCESSO!")
except Exception as e:
    print("❌ Modello Base FALLITO:", e)

print("\n3. Test Librerie C++ (Torch Geometric)...")
try:
    import torch_cluster
    from importlib.metadata import version

    print(f"torch_cluster version: {version('torch_cluster')}")
    # Creiamo due tensori dummy per forzare torch_cluster a usare la GPU
    pos = torch.rand((100, 3)).cuda()
    batch = torch.zeros(100, dtype=torch.long).cuda()
    fps_idx = torch_cluster.fps(pos, batch, ratio=0.5)
    print("✅ Torch Cluster C++ SUCCESSO!")
except Exception as e:
    print("❌ Torch Cluster C++ FALLITO:", e)
