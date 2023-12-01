import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the necessary modules for DDP and FSDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from fsdp import FullyShardedDataParallel

# Simple GPT-2 model
class GPT2(nn.Module):
    def __init__(self, vocab_size=10000, embed_size=256, num_heads=8, num_layers=6):
        super(GPT2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.fc_out(x)
        return x

# Training loop for a single GPU
def train_single_gpu(model, train_loader, criterion, optimizer, device):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# Training loop for Distributed Data Parallel (DDP)
def train_ddp(model, train_loader, criterion, optimizer, device, world_size, rank):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = DistributedDataParallel(model)
    model.to(device)

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# Training loop for Fully Sharded Data Parallel (FSDP)
def train_fsdp(model, train_loader, criterion, optimizer, device):
    model = FullyShardedDataParallel(model)
    model.to(device)

    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# Instantiate the model, loss, optimizer, and data loader
model = GPT2()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample data and targets
data = torch.randint(0, 10000, (1000, 512))
targets = torch.randint(0, 10000, (1000,))
train_dataset = torch.utils.data.TensorDataset(data, targets)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training parameters
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
rank = 0  # Set to the appropriate rank in DDP

# Choose the appropriate training loop based on the setup
distributed_type = "single_gpu"  # Choose 'single_gpu', 'ddp', or 'fsdp'
if distributed_type == "single_gpu":
    train_single_gpu(model, train_loader, criterion, optimizer, device)
elif distributed_type == "ddp":
    train_ddp(model, train_loader, criterion, optimizer, device, world_size, rank)
elif distributed_type == "fsdp":
    train_fsdp(model, train_loader, criterion, optimizer, device)
else:
    raise ValueError("Invalid distributed_type. Choose 'single_gpu', 'ddp', or 'fsdp'.")
