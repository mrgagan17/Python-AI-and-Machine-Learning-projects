import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from gan_architecture import Generator, Discriminator

# 1. Hyperparameters
batch_size = 128
image_size = 64
latent_dim = 100
num_epochs = 5  # Increase to 20-50 for high-quality faces, if you have a GPU
lr = 0.0002
beta1 = 0.5

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Windows Multiprocessing Protection Block
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load and Preprocess Dataset
    dataset = dset.ImageFolder(root="dataset",
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # Set num_workers=0 to prevent Windows spawning crashes
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3. Initialize Networks and Weights
    netG = Generator().to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # 4. Loss Function and Optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    # 5. Training Loop
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            
            # --- Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))) ---
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            # Forward pass real batch
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            # Generate fake image batch
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            # Forward pass fake batch
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            optimizerD.step()

            # --- Update Generator: maximize log(D(G(z))) ---
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD_real.item() + errD_fake.item():.4f} Loss_G: {errG.item():.4f}")

    # 6. Save the trained weights
    torch.save(netG.state_dict(), 'generator_weights.pth')
    print("Training Complete. Weights saved to generator_weights.pth")