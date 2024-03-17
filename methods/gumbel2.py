import torch
import torch.optim as optim 
import torch.nn as nn
import itertools


class FairLinear(nn.Module):
    def __init__(self, dim, num_envs):
        super().__init__()
        self.num_envs = num_envs
        self.g = nn.Linear(dim, 1)
        self.f = []
        for _ in range(num_envs):
            self.f.append(nn.Linear(dim, 1))
    
    def forward(self, x, pred=False):
        if pred:
            return self.g(x)
        else:
            out = []
            for i in range(self.num_envs):
                out.append(self.f[i](x))

def sample_gumbel(shape, device, eps=1e-20):
    # torch.manual_seed(seed)
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid(logits, temperature, grad=False):
    gumbel_softmax_sample = logits \
                            + sample_gumbel(logits.shape, logits.device) \
                            - sample_gumbel(logits.shape, logits.device)
    y = torch.sigmoid(gumbel_softmax_sample / temperature)
    if grad: # gradient of soft gumbel wrt. grad
        y_ = torch.sigmoid(-gumbel_softmax_sample / temperature)
        g = y * y_ / temperature
        g.fill_diagonal_(0)
        return y, g
    else:
        return y
    
def train(x, y, tau, gamma=1, lr=1e-3, lr_lam=1e-3, num_epochs=500, ini_value=0.0001, 
          step_d=2,
          device=torch.device("cpu"), print_every=10):
    # build models
    num_envs = len(x)
    dim = x[0].shape[1]
    model = FairLinear(dim, num_envs).to(device)
    opt_g = torch.optim.Adam(model.g.parameters(), lr=lr)
    opt_f = torch.optim.Adam(list(itertools.chain(*[model.f[i].parameters() for i in range(num_envs)])), lr=lr)
    
    if torch.is_tensor(ini_value):
        lam = ini_value
    else:
        lam = torch.ones(dim, device=device)*ini_value
    lam.requires_grad = True
    opt_lam = torch.optim.Adam([lam], lr=lr_lam)
    lams = []
    Ms = []
    losses = []
    losses_pred = []
    losses_penalty = []
    
    model.train()
    if not torch.is_tensor(x[0]):
        x = [torch.Tensor(x[i]) for i in range (num_envs)]
        y = [torch.Tensor(y[i]) for i in range (num_envs)]
    x = [x[i].to(device) for i in range(num_envs)]
    y = [y[i].to(device) for i in range(num_envs)]
    for idx_epoch in range(num_epochs):
        opt_lam.zero_grad()
        model.zero_grad()
                
        # update f
        for _ in range(step_d):
            M = gumbel_sigmoid(lam, temperature=tau).unsqueeze(0)
            model.zero_grad()
            pred_f = [model.f[i](M.detach()*x[i]) for i in range(num_envs)]
            pred_g = [model.g(M.detach()*x[i]).detach() for i in range(num_envs)]
            loss_f = -gamma * sum([((y[i] - pred_g[i]) * pred_f[i] - pred_f[i]**2 / 2).mean() for i in range(num_envs)])
            loss_f.backward()
            opt_f.step()
        
        # update g and lam
        model.zero_grad()
        M = gumbel_sigmoid(lam, temperature=tau).unsqueeze(0)
        pred_f = [model.f[i](M*x[i]) for i in range(num_envs)]
        pred_g = [model.g(M*x[i]) for i in range(num_envs)]
        #print(y[0].shape, pred_g[0].shape)
        loss_pred = 0.5 * sum([(y[i] - pred_g[i]).pow(2).mean() for i in range(num_envs)])
        loss_penalty = sum([((y[i] - pred_g[i]) * pred_f[i] - pred_f[i]**2 / 2).mean() for i in range(num_envs)])
        loss = loss_pred + gamma * loss_penalty
        loss.backward()
        opt_g.step()
        opt_lam.step()
        
        lams.append(lam.detach().cpu().tolist())
        Ms.append(M.detach().cpu().tolist())
        losses.append(loss.item())
        losses_pred.append(loss_pred.item())
        losses_penalty.append(loss_penalty.item())
        
        if idx_epoch % print_every == 0:
            print(f'model g = {model.g.weight.detach().cpu()}')
            print(f'model f1 = {model.f[0].weight.detach().cpu()}')
            print(f'model f2 = {model.f[1].weight.detach().cpu()}')
            print(idx_epoch, 'lam:', lam.detach().cpu().tolist(), '\n\t', loss_pred.item(), loss_penalty.item(), loss.item())
        
    losses_out = {
        "lam": lams,
        "M": Ms,
        "loss": losses,
        "loss_pred": losses_pred,
        "loss_penalty": losses_penalty
    }
        
    return model, losses_out