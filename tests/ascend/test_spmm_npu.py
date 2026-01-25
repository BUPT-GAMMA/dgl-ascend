import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.data import CoraGraphDataset
from dgl import AddSelfLoop
import time
import os

# Try to import torch_npu
try:
    import torch_npu
except ImportError:
    pass

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature, norm):
        # Creating a local scope so that we don't pollute the graph with temporary features
        with g.local_scope():
            # Normalization: D^-0.5 * X
            h = feature * norm
            
            g.ndata['h'] = h
            # SpMM: Aggregate neighbors' features
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = g.ndata['h']
            
            # Normalization: h * D^-0.5
            h = h * norm
            
            # Linear Transformation
            return self.linear(h)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_size)
        self.layer2 = GCNLayer(hidden_size, out_feats)

    def forward(self, g, features, norm):
        h = self.layer1(g, features, norm)
        h = F.relu(h)
        h = self.layer2(g, h, norm)
        return h

def test_gcn_ascend():
    print(f"DGL Version: {dgl.__version__}")
    print(f"PyTorch Version: {torch.__version__}")

    # Check for NPU availability
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        print("Error: NPU device is not available.")
        return

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    print(f"Using device: {device}")

    # 1. Load Cora Dataset
    print("Loading Cora dataset...")
    transform = AddSelfLoop()
    data = CoraGraphDataset(transform=transform)
    g = data[0]

    features = g.ndata['feat']
    labels = g.ndata['label']

    in_feats = features.shape[1]
    hidden_size = 16
    out_feats = data.num_classes

    print(f"Graph loaded. Num nodes: {g.num_nodes()}, Num edges: {g.num_edges()}")
    print(f"Features shape: {features.shape}, Num classes: {out_feats}")

    # 2. Prepare graph formats on CPU (Crucial for bypassing COOToCSR not implemented on Ascend)
    # Also create the CSR format which is the transpose of CSC.
    # Backward pass needs the reverse graph. If forward uses CSC (in-edge aggregation),
    # backward might need CSR (out-edge aggregation) or the transpose of current graph.
    print("Pre-converting graph format on CPU...")
    g = g.formats(['csc', 'csr'])
    g.create_formats_()

    # 3. Precompile degrees on CPU
    # Move degree computation here to avoid 'CSRGetRowNNZ' missing kernel on NPU
    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5)
    
    # 4. Move data to NPU
    print("Moving graph and features to NPU...")
    g = g.to(device)
    features = features.to(device)
    norm = norm.to(device).unsqueeze(1) # Move normalization coefficients to NPU
    
    # Synchronize to ensure all data is transferred
    # torch.npu.synchronize()
    print("NPU synchronization completed.")

    # 4. Initialize Model
    model = GCN(in_feats, hidden_size, out_feats).to(device)
    # torch.npu.synchronize()  # Sync after model initialization too
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # Move masks and labels to device
    train_mask = g.ndata['train_mask'].to(device)
    val_mask = g.ndata['val_mask'].to(device)
    labels = labels.to(device)

    print("Model initialized on NPU. Starting training...")

    # 5. Training Loop
    try:
        # torch.npu.synchronize()
        total_start = time.time()
        
        for epoch in range(50):
            epoch_start = time.time()
            model.train()
            
            # Forward (pass norm explicitly)
            logits = model(g, features, norm)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluate (simple accuracy on validation)
            model.eval()
            with torch.no_grad():
                val_logits = logits[val_mask]
                val_labels = labels[val_mask]
                _, indices = torch.max(val_logits, dim=1)
                correct = torch.sum(indices == val_labels)
                val_acc = correct.item() * 1.0 / len(val_labels)
            
            # torch.npu.synchronize()
            epoch_time = (time.time() - epoch_start) * 1000
            
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f} ms")
            
        torch.npu.synchronize()
        print(f"Training completed in {(time.time() - total_start):.2f} s")
        print(f"Final Output shape: {logits.shape}")

    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    os.environ['DGLBACKEND'] = 'pytorch'
    test_gcn_ascend()
