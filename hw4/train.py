import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def train_conditionGAN(G, D, optim_G, optim_D, loss_f, train_loader, num_epochs, device):

    for epoch in range(num_epochs):

        for i, (img, number) in enumerate(train_loader):
            batch_size = img.shape[0]

            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)

            # ========================
            #   Train Discriminator
            # ========================
            # train with real data
            img = img.to(device)
            number = number.to(device)
            real_score = D(img, number)
            d_loss_real = loss_f(real_score, real_label)

            # train with fake data
            noise = torch.randn(batch_size, 100, device=device)
            img_fake = G(noise, number)
            fake_score = D(img_fake, number)
            d_loss_fake = loss_f(fake_score, fake_label)

            # update D
            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # ========================
            #   Train Generator
            # ========================
            noise = torch.randn(batch_size, 100, device=device)
            fake_number = torch.randint(0, 10, (batch_size,), device=device)
            img_fake = G(noise, fake_number)
            g_score = D(img_fake, fake_number)
            g_loss = loss_f(g_score, real_label)

            # update G
            G.zero_grad()
            g_loss.backward()
            optim_G.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item(),
                         real_score.mean().item(), fake_score.mean().item(), g_score.mean().item()))

        noise = torch.randn(50, 100, device=device)
        fake_number = torch.arange(0, 10, device=device)[None].repeat(5, 1).view(-1).long()
        img_fake = G(noise, fake_number)
        grid = make_grid(img_fake, nrow=10)
        plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()