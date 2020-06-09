#!/usr/bin/env python3

# from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ignite.engine import Events, create_supervised_trainer
from ignite.engine import create_supervised_evaluator


BATCH_SIZE = 128
LOG_INTERVAL = 50
EPOCHS = 10

torch.manual_seed(1)
# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == 'cuda'


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    # Reconstruction + KL divergence losses summed over all elements and batch
    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

def test():
    low = torch.tensor([1,2,3,4,5,6])
    target = torch.zeros([4,4])
    low
    target
    ct=0
    for i in range(4):
        for j in range(i):
            print(i, j, ct)
            target[i][j] = low[ct]
            ct+=1
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    a = torch.tensor([1,2])
    b = torch.tensor([[1],[2]])
    b = torch.tensor([1,2])
    a.shape
    torch.matmul(a,b)
    mat1.shape
    mat2.shape
    torch.mm(mat1, mat2)    

class CausalVAE(nn.Module):
    def __init__(self):
        super(CausalVAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        # FIXME most of the weights are not used, but would it create
        # overhead?
        self.fc23 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
    def encode(self, x):
        # x = torch.ones([784])
        # x.shape
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        sigma = self.fc22(h1)
        C = self.fc23(h1)
        print(C.shape)
        z_C = torch.tril(C.view(-1,20,20), diagonal=-1)
        # torch.eye(20).shape
        # C = torch.ones([400])
        # C.view([20,20]).shape
        mat_left = torch.inverse(torch.eye(20) - z_C)
        mat_right = torch.t(mat_left)
        mat_middle = torch.diag(torch.dot(sigma, sigma))
        z_mu = torch.matmul(torch.matmul(mat_left, mat_middle), mat_right)
        return z_C, z_mu, z_sigma
    
    def reparameterize(self, C, mu, sigma):
        z = torch.inverse(torch.eye(20) - C)
        z = torch.matmul(z, sigma)
        # N(0,1)
        eps = torch.randn_like(mu)
        z = mu + torch.matmul(z, eps)
        return z
 
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        C, mu, sigma = self.encode(x.view(-1, 784))
        z = self.reparameterize(C, mu, sigma)
        return self.decode(z), C, mu, sigma
    # Reconstruction + KL divergence losses summed over all elements and batch
    @staticmethod
    def loss_function(recon_x, x, C, mu, sigma):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        term1 = torch.log(torch.det(sigma))
        term2 = torch.trace(sigma)
        term3 = torch.dot(mu, mu)
        KLD = -0.5 * torch.sum(1 + term1 - term2 - term3)
        return BCE + KLD


def use_ignite():
    assert False
    def update_model(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, _ = prepare_batch(batch)
        recon_x, C, mu, sigma = model(x)
        loss = VAE.loss_function(recon_x, x, C, mu, sigma)
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(update_model)
    trainer.run(data, max_epochs=100)
    trainer = create_supervised_trainer(model, optimizer, loss)
    return
    

def load_mnist_data():
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    return train_loader, test_loader

def train_vae(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = VAE.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test_vae(epoch, test_loader):
    model.eval()
    test_loss = 0
    if not os.path.exists("results"):
        os.makedirs("results")
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += VAE.loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main_vae():
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader = load_mnist_data()
    for epoch in range(1, EPOCHS + 1):
        train_vae(epoch, model, optimizer, train_loader)
        test_vae(epoch, test_loader)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')






def train_causal_vae(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, C, mu, sigma = model(data)
        loss = VAE.loss_function(recon_batch, data, C, mu, sigma)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test_causal_vae(epoch, test_loader):
    model.eval()
    test_loss = 0
    if not os.path.exists("results"):
        os.makedirs("results")
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, C, mu, sigma = model(data)
            test_loss += VAE.loss_function(recon_batch, data, C, mu, sigma).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
def causal_vae_main():
    model = CausalVAE().to(device)
    train_loader, test_loader = load_mnist_data()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    for epoch in range(1, 10 + 1):
        train_causal_vae(epoch, model, optimizer, train_loader)
        test_causal_vae(epoch, test_loader)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'causal-results/sample_' + str(epoch) + '.png')
