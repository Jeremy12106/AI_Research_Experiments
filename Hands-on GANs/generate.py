
import os
import torch
import matplotlib.pyplot as plt
import torchvision
from GANs_training import Generator  # 假設你將Generator定義放在model.py中
from torchvision.utils import make_grid

# 設定生成圖像的參數
latent_dim = 128  # 生成器的隱含向量維度
image_size = 128  # 圖像大小
image_channels = 3  # 圖像通道數（RGB = 3）
num_classes = 6

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載生成器模型
generator = Generator(latent_dim, num_classes, image_size, image_channels).to(device)

# 加載已訓練的權重
generator.load_state_dict(torch.load('results/generator_final.pth'))
generator.eval()  # 將模型設為評估模式

# 生成圖像的函數
def generate_images(generator, latent_dim, num_classes, num_images=10, nrow=5, output_dir='generated_images'):
    os.makedirs(output_dir, exist_ok=True)  # 確保輸出目錄存在

    with torch.no_grad():
        # 隨機生成噪聲向量
        z = torch.randn(num_images, latent_dim, device=device)
        
        # 隨機生成對應的類別標籤
        labels = torch.randint(0, num_classes, (num_images,), device=device)

        # 使用生成器生成圖像
        generated_imgs = generator(z, labels)

        # 重新縮放到 [0, 1]
        generated_imgs = (generated_imgs + 1) / 2

        # 生成網格圖像並保存
        grid = make_grid(generated_imgs.cpu(), nrow=nrow, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(f'{output_dir}/generated.png', bbox_inches='tight', pad_inches=0.1)
        print(f'Generated images saved to {output_dir}/generated.png')

# 主函數
if __name__ == '__main__':
    # 生成10張圖像，並按類別保存
    generate_images(generator, latent_dim, num_classes, num_images=10, nrow=5)
