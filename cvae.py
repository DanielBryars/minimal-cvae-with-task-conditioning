import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import datetime
from tqdm import tqdm 

#Something I learn't at MLX - determinstic training for reproducibility.
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_modal_data(lChunk = 10, nChunks=2000, modes=[1.0, -1.0], noise_std=0.05):

    chunks = []    
    #unpytonic way but it's clear what's happening
    for _ in range(0, nChunks):
        mode = random.choice(modes) # pick the mode for this chunk
        chunk = torch.normal(mean=mode, std=noise_std, size=(lChunk,)) #mode = mean of gaussian        
        chunks.append(chunk)

    x = torch.rand(nChunks, 1) * 2 - 1 # uniform random data in [-1, 1]
    
    chunksTensor = torch.stack(chunks) # shape (nChunks, lChunk)
    return x, chunksTensor
    
class TinyMlp(nn.Module):
    def __init__(self, embedding_size = 64, output_size = 10):
        super().__init__()

        self.pipeline = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, output_size)
        )
        
    def forward(self, x):
        return self.pipeline(x)

def train_and_eval_tiny_mlp(x, chunks):
    dataset = torch.utils.data.TensorDataset(x, chunks)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    print(f"Using device:{device}")

    model = TinyMlp()
    model.to(device)
    print('model:params', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-3)
    
    num_epochs = 100
    
    for epoch in range(1, num_epochs + 1):
        loop = tqdm(
            dataloader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
            dynamic_ncols=False,
            ncols=100,
        )
        for batch in loop:
            batch_x, batch_chunks = [d.to(device) for d in batch]

            predictions =  model(batch_x)
            loss = F.mse_loss(predictions, batch_chunks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix_str(f"loss={loss.item():10.4f}")

    #eval, should give a collapsed mean of the modes
    test_x = torch.tensor([[0.0], [0.5], [-0.5]]).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(test_x).cpu()
        for i in range(3):
          print(f"x={test_x[i].item():.1f}  chunk_mean={predictions[i].mean().item():.3f}")


if __name__ == "__main__":
    set_seed()

    x, chunks = make_modal_data()
    #for i in range(10):
    #  print(f"chunk_mean={chunks[i].mean().item():.2f}")

    #print(chunks.shape)
    #print(chunks)
    
    #scratch()

    train_and_eval_tiny_mlp(x, chunks)

    print("Letz go")