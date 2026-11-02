import os
import sys
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

if len(sys.argv) < 2:
    print("Usage: python3 cnn_weight_image_generator.py <bit_width> [seed]")
    sys.exit(1)

TARGET_BITS = int(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NUM_SAMPLES = 1000

MAX_SIGNED_VAL = min((1 << (TARGET_BITS - 1)) - 1, 10000)
QAT_SCALE = float(MAX_SIGNED_VAL)
DATA_DIR = f"mnist_cnn_data_{TARGET_BITS}"

# Reproducibility: same seed -> same trained network every time
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class FakeQuantizeSafe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        clamped_x = torch.clamp(x, -1.0, 1.0)
        return torch.round(clamped_x * QAT_SCALE) / QAT_SCALE

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def qat_safe(x):
    return FakeQuantizeSafe.apply(x)


def normalize_per_sample(x):
    """Divide each sample by its OWN max (same as the inference scripts do).
    The old code divided by the max of the whole batch, which is a different
    network from the one used at inference time."""
    x_max = x.amax(dim=tuple(range(1, x.dim())), keepdim=True)
    return x / x_max.clamp(min=1e-12)


class ModernCustomCNN_QAT(nn.Module):
    def __init__(self):
        super(ModernCustomCNN_QAT, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(8 * 6 * 6, 10)

    def forward(self, x):
        x = qat_safe(x)
        w1_qat = qat_safe(self.conv1.weight)
        b1_qat = qat_safe(self.conv1.bias)
        x = self.relu1(nn.functional.conv2d(x, w1_qat, b1_qat, stride=2))

        x = normalize_per_sample(x)
        x = qat_safe(x)

        w2_qat = qat_safe(self.conv2.weight)
        b2_qat = qat_safe(self.conv2.bias)
        x = self.relu2(nn.functional.conv2d(x, w2_qat, b2_qat, stride=2))

        x = normalize_per_sample(x)
        x = x.view(x.size(0), -1)
        x = qat_safe(x)

        w3_qat = qat_safe(self.fc.weight)
        b3_qat = qat_safe(self.fc.bias)
        x = nn.functional.linear(x, w3_qat, b3_qat)
        return x


def export_matrix(filename, matrix, is_integer=False):
    fmt = '%d' if is_integer else '%.6f'
    np.savetxt(filename, matrix, fmt=fmt)


def to_float(a):
    """Float export, clamped to [-1, 1] = exactly what the QAT forward pass used."""
    return np.clip(a, -1.0, 1.0)


def to_int(a):
    """Integer export, clipped to [-S, S] so it always fits in TARGET_BITS (2's complement)."""
    out = np.clip(np.round(a * QAT_SCALE), -MAX_SIGNED_VAL, MAX_SIGNED_VAL).astype(np.int64)
    assert out.min() >= -(1 << (TARGET_BITS - 1)) and out.max() <= (1 << (TARGET_BITS - 1)) - 1, \
        "Exported integer does not fit in the requested bit width"
    return out


if __name__ == "__main__":
    # Always start from an empty folder so stale sample_img_*_label_*.txt files can never survive
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data', train=False, download=True, transform=transform)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, generator=g)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    model = ModernCustomCNN_QAT()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print(f"Training Safe QAT CNN ({TARGET_BITS}-bit, scale={MAX_SIGNED_VAL}, seed={SEED}) for 5 epochs...")
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # Keep parameters inside the range the fake-quantizer can represent,
            # so the exported weights are the weights that were actually trained.
            with torch.no_grad():
                for p in model.parameters():
                    p.clamp_(-1.0, 1.0)
            running_loss += loss.item()
        print(f"   Epoch {epoch+1}/5 Complete | Loss: {running_loss/len(train_loader):.4f}")

    model.eval()

    # Sanity check: accuracy of the quantized network on the full MNIST test set
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            correct += (model(data).argmax(dim=1) == target).sum().item()
    print(f"   QAT model test accuracy: {correct / len(test_set) * 100:.2f}%")

    # Layer 1 Export
    w1 = model.conv1.weight.detach().numpy()
    b1 = model.conv1.bias.detach().numpy()
    w1_flat = w1.reshape(w1.shape[0], -1).T
    export_matrix(f"{DATA_DIR}/W1_float.txt", to_float(w1_flat))
    export_matrix(f"{DATA_DIR}/B1_float.txt", to_float(b1))
    export_matrix(f"{DATA_DIR}/W1_matrix.txt", to_int(w1_flat), True)
    export_matrix(f"{DATA_DIR}/B1_vector.txt", to_int(b1), True)

    # Layer 2 Export
    w2 = model.conv2.weight.detach().numpy()
    b2 = model.conv2.bias.detach().numpy()
    w2_flat = w2.reshape(w2.shape[0], -1).T
    export_matrix(f"{DATA_DIR}/W2_float.txt", to_float(w2_flat))
    export_matrix(f"{DATA_DIR}/B2_float.txt", to_float(b2))
    export_matrix(f"{DATA_DIR}/W2_matrix.txt", to_int(w2_flat), True)
    export_matrix(f"{DATA_DIR}/B2_vector.txt", to_int(b2), True)

    # Layer 3 (FC) Export
    w3 = model.fc.weight.detach().numpy()
    b3 = model.fc.bias.detach().numpy()
    w3_flat = w3.T
    export_matrix(f"{DATA_DIR}/W3_float.txt", to_float(w3_flat))
    export_matrix(f"{DATA_DIR}/B3_float.txt", to_float(b3))
    export_matrix(f"{DATA_DIR}/W3_matrix.txt", to_int(w3_flat), True)
    export_matrix(f"{DATA_DIR}/B3_vector.txt", to_int(b3), True)

    # Export the first NUM_SAMPLES TEST images in fixed order (same 100 images for every
    # bit width and every HRS value; not training images)
    for i in range(NUM_SAMPLES):
        img_tensor, label = test_set[i]
        img_2d = img_tensor.numpy().squeeze()
        np.savetxt(f"{DATA_DIR}/sample_img_{i}_float.txt", img_2d, fmt='%.6f')
        np.savetxt(f"{DATA_DIR}/sample_img_{i}_label_{int(label)}.txt",
                   np.round(img_2d * QAT_SCALE).astype(np.int64), fmt='%d')

    print(f"Export Complete: CNN weights unrolled and {NUM_SAMPLES} test images saved to {DATA_DIR}/")
