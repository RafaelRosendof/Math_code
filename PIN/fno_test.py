import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SpecConv1d(nn.Module):
    def __init__(self, in_chan, out_chan, modes):
        super(SpecConv1d, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.modes = modes 
        self.scale = (1 / (in_chan * out_chan))
        self.weights = nn.Parameter(self.scale * torch.rand(in_chan, out_chan, modes, dtype=torch.cfloat))
        
        
    def complex_mult(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x) # [BATCH , CHANNESL, GRID]
        
        out_ft = torch.zeros(batch_size, self.in_chan, x.size(-1)//2, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.complex_mult(x_ft[:, :, :self.modes], self.weights)
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    
class FNO(nn.Module):
    def __init__(self, modes, width):
        super(FNO, self).__init__()
        
        self.fc0 = nn.Linear(2, width)
        # (u(x), x)
        self.conv0 = SpecConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0,2,1)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        x = x.permute(0,2,1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
def generate_data(num_samples, grid_size=1000):
    
    x_grid = torch.linspace(0, 1, grid_size)
    
    inputs = []
    outs = []
    
    for _ in range(num_samples):
        split = torch.randint(20, 80, (1,)).item()
        u0 = torch.zeros(grid_size)
        u0[:split] = 1.0 # hot liquid
        u0[split:] = 0.0
        
        sigma = 5.0
        kernel = torch.exp(-torch.linspace(-10, 10, grid_size) ** 2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        u_t = F.conv1d(u0.view(1,1,-1), kernel.view(1,1,-1), padding='same').view(-1)
        inp = torch.stack([u0, x_grid], dim=1)

        inputs.append(inp)
        outs.append(u_t.unsqueeze(1))
        
    return torch.stack(inputs), torch.stack(outs)

train_x, train_y = generate_data(1000) # 1000 samples
test_x, test_y = generate_data(10)

# Init Model
modes = 16  # Keep 16 low-frequency modes
width = 64  # Hidden layer width
model = FNO(modes, width)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Training FNO...")
for epoch in range(51):
    optimizer.zero_grad()
    pred = model(train_x)
    loss = criterion(pred, train_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.6f}")
    
    
with torch.no_grad():
    sample_idx = 0
    pred = model(test_x[sample_idx:sample_idx+1])
    
    # Extract data for plotting
    initial_state = test_x[sample_idx, :, 0].numpy() # The sharp step
    true_final = test_y[sample_idx, :, 0].numpy()    # The smooth diffusion
    pred_final = pred[0, :, 0].numpy()               # FNO prediction

    # Simple ASCII Plot since I cannot render images directly
    print("\n--- Visual Check (Middle of grid) ---")
    print(f"Initial (Hot/Cold Boundary): {initial_state[50]:.1f}") 
    print(f"True Final Temp:             {true_final[50]:.4f}")
    print(f"FNO Predicted Temp:          {pred_final[50]:.4f}")
    print("-------------------------------------")
    print("If 'Predicted' is close to 'True', the FNO learned the diffusion operator!")