
# coding: utf-8

# In[572]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

from random import randint

from IPython.display import Image
from IPython.core.display import Image, display

# %load_ext autoreload
# %autoreload 2


# In[573]:


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[574]:


bs = 64


# In[575]:


# Load Data
dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
len(dataset)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
print (len(data_loader))


# In[576]:


fixed_x, _ = next(iter(data_loader))
# check first batch data
torchvision.utils.save_image(fixed_x, 'real_image.png' )
Image('real_image.png')


# In[577]:


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# In[578]:


class UnFlatten(nn.Module):
    def forward(self, input, size=192):
        return input.view(input.size(0), size, 1, 1)


# In[579]:


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=192, z_dim=14):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=5, stride=1),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


# In[580]:


image_channels = fixed_x.size(1)


# In[581]:


vae = VAE(image_channels=image_channels).to(device)
# model.load_state_dict(torch.load('vae.torch', map_location='cpu'))


# In[582]:


optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3) 


# In[583]:


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


# In[584]:


get_ipython().system('rm -rfr reconstructed')
get_ipython().system('mkdir reconstructed')


# In[591]:


epochs = 1


# In[592]:


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# In[593]:


def flatten(x):
    return to_var(x.view(x.size(0), -1))


# In[594]:


for epoch in range(epochs):
    for idx, (images, _) in enumerate(data_loader):
        recon_images, mu, logvar = vae(images)
        loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx%100 == 0:
            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data[0]/bs))
            recon_x, _, _ = vae(fixed_x)
            save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), f'reconstructed/recon_image_{epoch}_{idx}.png')


# In[ ]:


def compare(x):
    recon_x, _, _ = vae(x)
    return torch.cat([x, recon_x])


# In[595]:


sample = torch.randn(bs, 192)
compare_x = vae.decoder(sample)

save_image(compare_x.data.cpu(), 'sample_image.png')
display(Image('sample_image.png', width=700, unconfined=True))


# In[596]:


fixed_x = dataset[randint(1, 100)][0].unsqueeze(0)
compare_x = compare(fixed_x)

save_image(compare_x.data.cpu(), 'sample_image.png')
display(Image('sample_image.png', width=700, unconfined=True))

