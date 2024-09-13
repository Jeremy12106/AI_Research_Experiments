import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 自定義數據集類別，用於讀取圖像和標籤
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self.inverse_label_map = {}

        try:
            label_dirs = os.listdir(root_dir)
            for idx, label_dir in enumerate(label_dirs):
                self.label_map[label_dir] = idx
                self.inverse_label_map[idx] = label_dir
                label_path = os.path.join(root_dir, label_dir)
                if os.path.isdir(label_path):
                    for img_file in os.listdir(label_path):
                        if img_file.endswith('.jpg'):
                            self.image_paths.append(os.path.join(label_path, img_file))
                            self.labels.append(idx)
        except FileNotFoundError:
            print(f"Error: Directory {root_dir} not found.")
            raise

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# 生成器模型定義
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, image_size, image_channels):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = image_size // 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 256 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, image_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# 判别器模型定義        
class Discriminator(nn.Module):
    def __init__(self, num_classes, image_size, image_channels):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Conv2d(image_channels + num_classes, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512 * (image_size // 16) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embeddings = self.label_embedding(labels).view(labels.size(0), -1, 1, 1).repeat(1, 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label_embeddings), 1)
        validity = self.model(d_in)
        return validity

if __name__ == '__main__':
    # 檢查設備（CUDA 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用設備: {device}')

    # 參數設定
    latent_dim = 128
    image_size = 128
    image_channels = 3
    batch_size = 64
    epochs = 101
    learning_rate = 0.0002
    beta1 = 0.5

    num_classes = len(os.listdir('assets/datasets'))

    # 定義圖像轉換
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 建立數據集和數據加載器
    dataset = CustomDataset(root_dir='assets/datasets', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    generator = Generator(latent_dim, num_classes, image_size, image_channels).to(device)
    discriminator = Discriminator(num_classes, image_size, image_channels).to(device)

    # 優化器
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # 損失函數
    adversarial_loss = nn.BCELoss()

    d_losses = []
    g_losses = []

    os.makedirs('results', exist_ok=True)

    # 訓練模型
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            
            ### 訓練生成器 ###
            batch_size = imgs.size(0)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)

            ### 訓練判別器 ###
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            print(f'Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} '
                  f'Loss D: {d_loss.item()}, Loss G: {g_loss.item()}')
        
        if epoch % 10 == 0:
            with torch.no_grad():
                num_imgs = 64
                nrow = 8
                z = torch.randn(num_imgs, latent_dim, device=device)
                gen_labels = torch.randint(0, num_classes, (num_imgs,), device=device)
                gen_imgs = generator(z, gen_labels)

                # 重新縮放到 [0, 1]
                gen_imgs = (gen_imgs + 1) / 2

                # 生成網格
                grid = torchvision.utils.make_grid(gen_imgs.cpu(), nrow=nrow, normalize=True)

                # 保存圖片
                plt.imshow(grid.permute(1, 2, 0).numpy())
                plt.axis('off')
                plt.savefig(f'results/epoch_{epoch}.png', bbox_inches='tight', pad_inches=0.1)
                plt.close()

    # 繪製損失值
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('GAN Loss During Training')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/gans_training_loss.png')
    plt.show()
    plt.close()
