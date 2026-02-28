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

#(venv) PS E:\git\ai\minimal-cvae-with-task-conditioning> python .\cvae.py
#  Releasing the hounds
#  Using device:cpu
#  model:params 4938
#  x=0.0  chunk_mean=-0.047
#  x=0.5  chunk_mean=-0.046
#  x=-0.5  chunk_mean=-0.009
#  Letz go

# These means are all about 0, half way beedn the two modes of 1 and -1, 
# which is what we expect from a model that has collapsed the modes together. The model has no incentive to learn to separate the modes, so it learns to predict the mean of the modes, which is 0. 

################################################################### CVAE

# q(z | x, chunk)
class Encoder(nn.Module):
    def __init__(self, x_dim = 1, chunk_dim = 10, embedding_size = 64):
        super().__init__()

        self.pipeline = nn.Sequential(
            nn.Linear(x_dim + chunk_dim, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 2)

             # mu and log_var — the KL loss encourages these toward N(0,1),
            # while reconstruction loss pushes them to encode which mode this sample belongs to
        )
        
    def forward(self, x, chunk):
        # always batched, even if batch size is 1, so x and chunk should both be 2D tensors with shape (batch_size, feature_dim)
        if x.ndim != 2 or chunk.ndim != 2 or x.size(0) != chunk.size(0): 
            raise ValueError(f"incompatible shapes: x{tuple(x.shape)} chunk{tuple(chunk.shape)}")

        output = self.pipeline(torch.cat([x, chunk], dim=-1))

        mu, log_var = output[:, 0:1], output[:, 1:2]
        return mu, log_var


#Decoder p(chunk | x, z): takes x and a sampled z, outputs predicted chunk
#  [x(1), z(1)] → 64 → ReLU → 64 → ReLU → chunk(10)
class Decoder(nn.Module):
    def __init__(self, x_dim = 1, z_sampled_dim=1, chunk_dim = 10, embedding_size = 64):
        super().__init__()

        self.pipeline = nn.Sequential(
            nn.Linear(x_dim + z_sampled_dim, embedding_size), #project x and z into embedding space
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size), #hidden layer
            nn.ReLU(),
            nn.Linear(embedding_size, chunk_dim) #project back out to chunk space, this is the reconstruction of the chunk
        )
        
    def forward(self, x,z_sampled):
        return self.pipeline(torch.cat([x, z_sampled], dim=-1)) #I've gone with z last, seems to match the ACT paper

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, chunk):
        mu, log_var = self.encoder(x, chunk)

        #reparameterization trick
        epsilon = torch.randn_like(log_var) 
        z_sampled = mu + torch.exp(0.5 * log_var) * epsilon 

        predicted_chunk = self.decoder(x, z_sampled)
        return predicted_chunk, mu, log_var

#################################

def train_and_eval_tiny_cvae(x, chunks):
    dataset = torch.utils.data.TensorDataset(x, chunks)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    print(f"Using device:{device}")

    model = CVAE()
    model.to(device)
    print('model:params', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-3)
    
    num_epochs = 100
    step = 1

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

            predictions, mu, log_var =  model(batch_x, batch_chunks)
            mse = F.mse_loss(predictions, batch_chunks)
            beta = step / (num_epochs * len(dataloader)) # linearly increase beta from 0 to 1 over the course of training

            kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()) #MAGIC

            loss = mse + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix_str(f"loss={loss.item():10.4f}")

            step += 1

    #eval, should give a collapsed mean of the modes
    test_x = torch.tensor([[0.0], [0.5], [-0.5]]).to(device)
    #test_x = torch.tensor([[0.0]]).to(device)

    model.eval()
    with torch.no_grad():
        #sampled_z = torch.zeros(test_x.size(0), 1).to(device) # use 0
        #z = torch.full((1, 1), z_val).to(device)          

        for z in [-2.0, -1.0, -0.0001, 0.0, 0.0001, 1.0, 2.0]:
        #for z in [0.0]:        
            sampled_z = torch.full((test_x.size(0), 1), z, device=device) # same z for all samples in the batch, so we can see how changing z changes the predictions
            predictions = model.decoder(test_x, sampled_z).cpu()  

            for i in range(test_x.size(0)):                              
                print(f"x={test_x[i].item():.1f}  chunk_mean={predictions[i].mean().item():.3f}")

# There's no signal between x and the chunks (different modes) and even with KL CVAE still collapses the modes together, predicting the mean of the modes. 
# So I think there's something fishy in the ACT paper.

#(venv) PS E:\git\ai\minimal-cvae-with-task-conditioning> python .\cvae.py
#Releasing the hounds
#Using device:cpu
#model:params 10060
#x=0.0  chunk_mean=0.008
#x=0.5  chunk_mean=-0.111
#x=-0.5  chunk_mean=0.024
#Letz go

if __name__ == "__main__":
    print("Releasing the hounds")
    set_seed()

    x, chunks = make_modal_data()
    #for i in range(10):
    #  print(f"chunk_mean={chunks[i].mean().item():.2f}")

    #print(chunks.shape)
    #print(chunks)
    
    #scratch()

    #train_and_eval_tiny_mlp(x, chunks)
    train_and_eval_tiny_cvae(x, chunks)
    print("Letz go")