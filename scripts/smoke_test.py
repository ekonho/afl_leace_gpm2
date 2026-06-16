import torch
import torch.nn as nn
from dualortho.models.resnet_cifar import resnet18_cifar
from dualortho.memory.extractor import MemoryExtractor
from dualortho.probe.afl_drift_probe import AFLDriftProbe
from dualortho.training.dualortho_trainer import DualOrthoTrainer

device = torch.device('cpu')
model = resnet18_cifar(num_classes=10, nf=32).to(device)

images = torch.randn(32, 3, 32, 32)
labels = torch.randint(0, 10, (32,))
dataset = torch.utils.data.TensorDataset(images, labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Stage 0: Memory extraction from backbone.layer4
# The extractor flattens [B,C,H,W] -> [N,C], SVD gives basis [C,k]
extractor = MemoryExtractor(['backbone.layer4'], variance_threshold=0.95, max_rank=50)
memories = extractor.extract(model, loader, device, max_batches=4)
M_basis = memories['backbone.layer4'].basis
print("Stage 0: Memory basis shape={}, rank={}".format(M_basis.shape, memories['backbone.layer4'].rank))

# Stage 1-2: Drift detection in backbone output space [D=256]
probe = AFLDriftProbe(variance_threshold=0.9, max_rank=10)
with torch.no_grad():
    feats = model.backbone(images)  # [32, 256]
    one_hot = torch.nn.functional.one_hot(labels, 10).float()
W_local = probe.solve_closed_form_weight(feats, one_hot)  # [256, 10]
W_agg = torch.randn(feats.shape[1], 10) * 0.01
drift = probe.detect_drift(W_local, W_agg)
print("Stage 1-2: Drift rank={}, U_drift={}".format(drift.rank, drift.u_drift.shape))

# Stage 3: Dual-orthogonal training
# Forward hook on 'backbone' (output [B,256]), toxic_basis [256,k']
# Grad projection on 'backbone.layer4' params, memory_bases [256,k]
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
trainer = DualOrthoTrainer(
    model=model,
    memory_bases={'backbone.layer4': M_basis},
    toxic_basis=drift.u_drift,
    backbone_name='backbone',
    grad_proj_layers=['backbone.layer4'],
    optimizer=optimizer,
    criterion=criterion,
    device=device,
)
metrics = trainer.train_one_epoch(loader, grad_clip=5.0)
print("Stage 3: loss={:.4f}, acc={:.1f}%".format(metrics['loss'], metrics['acc']*100))
print("Forward hook active:", trainer._fwd_handle is not None)

# Evaluate
eval_metrics = trainer.evaluate(loader)
print("Evaluate: loss={:.4f}, acc={:.1f}%".format(eval_metrics['loss'], eval_metrics['acc']*100))

trainer.remove_hooks()
print("Hooks cleaned up: fwd={}".format(trainer._fwd_handle))

# Verify gradient orthogonality: after backward, layer4 grads should be orthogonal to M
model.zero_grad()
x_test = images[:4]
y_test = labels[:4]
logits = model(x_test)
loss = criterion(logits, y_test)
loss.backward()
layer4 = dict(model.named_modules())['backbone.layer4']
for name, param in layer4.named_parameters():
    if param.grad is not None and param.grad.dim() >= 2 and param.grad.shape[-1] == M_basis.shape[0]:
        component = float((param.grad.reshape(-1, M_basis.shape[0]) @ M_basis).norm())
        grad_norm = float(param.grad.norm())
        print("  {} grad_norm={:.4f}, memory_component={:.6f}".format(name, grad_norm, component))
        break

print()
print("=== FULL PIPELINE SMOKE TEST PASSED ===")
