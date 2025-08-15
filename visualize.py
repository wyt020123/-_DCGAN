import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision.utils import make_grid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """训练可视化工具"""
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        self.losses = {
            'total_loss': [],
            'gan_loss': [],
            'recon_loss': [],
            'perceptual_loss': [],
            'style_loss': [],
            'd_loss': []
        }
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建时间戳文件夹
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_dir = os.path.join(save_dir, self.timestamp)
        os.makedirs(self.current_dir, exist_ok=True)
        
        # 创建子文件夹
        self.loss_plot_dir = os.path.join(self.current_dir, 'loss_plots')
        self.image_dir = os.path.join(self.current_dir, 'generated_images')
        os.makedirs(self.loss_plot_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def update_losses(self, losses_dict):
        """更新损失值"""
        for k, v in losses_dict.items():
            if k in self.losses:
                self.losses[k].append(v)

    def plot_losses(self, epoch):
        """绘制损失曲线"""
        plt.figure(figsize=(15, 10))
        
        # 绘制所有损失
        for loss_name, loss_values in self.losses.items():
            if loss_values:  # 确保有数据
                plt.plot(loss_values, label=loss_name)
        
        plt.title('Training Losses')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存损失图
        save_path = os.path.join(self.loss_plot_dir, f'losses_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()
        
        # 同时保存损失数据
        np.save(os.path.join(self.loss_plot_dir, f'losses_epoch_{epoch}.npy'), self.losses)
        
        logger.info(f"损失图已保存: {save_path}")

    def plot_separate_losses(self, epoch):
        """分别绘制每种损失"""
        for loss_name, loss_values in self.losses.items():
            if not loss_values:  # 跳过空数据
                continue
                
            plt.figure(figsize=(10, 6))
            plt.plot(loss_values, label=loss_name)
            plt.title(f'{loss_name} over time')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            save_path = os.path.join(self.loss_plot_dir, f'{loss_name}_epoch_{epoch}.png')
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"{loss_name}损失图已保存: {save_path}")

    def save_generated_images(self, images, epoch, n_images=16):
        """保存生成的图像"""
        # 确保图像数量不超过批次大小
        n_images = min(n_images, images.size(0))
        
        # 选择前n_images张图片
        images = images[:n_images]
        
        # 创建图像网格
        grid = make_grid(images, nrow=int(np.sqrt(n_images)), normalize=True)
        
        # 转换为numpy数组并调整通道顺序
        grid = grid.cpu().numpy().transpose(1, 2, 0)
        
        # 保存图像
        save_path = os.path.join(self.image_dir, f'generated_epoch_{epoch}.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        logger.info(f"生成的图像已保存: {save_path}")

    def plot_lr_schedule(self, optimizer, num_epochs):
        """绘制学习率调度曲线"""
        lrs = []
        for epoch in range(num_epochs):
            lrs.append([param_group['lr'] for param_group in optimizer.param_groups])
            optimizer.step()
        
        plt.figure(figsize=(10, 6))
        for i, lr in enumerate(zip(*lrs)):
            plt.plot(lr, label=f'Parameter Group {i}')
        
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.loss_plot_dir, 'learning_rate_schedule.png')
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"学习率调度图已保存: {save_path}")

    def generate_training_report(self, epoch, model_info=None):
        """生成训练报告"""
        report = f"Training Report - Epoch {epoch}\n"
        report += "=" * 50 + "\n\n"
        
        # 添加模型信息
        if model_info:
            report += "Model Information:\n"
            for k, v in model_info.items():
                report += f"{k}: {v}\n"
            report += "\n"
        
        # 添加损失信息
        report += "Loss Statistics:\n"
        for loss_name, loss_values in self.losses.items():
            if loss_values:
                report += f"{loss_name}:\n"
                report += f"  - Current: {loss_values[-1]:.4f}\n"
                report += f"  - Mean: {np.mean(loss_values):.4f}\n"
                report += f"  - Min: {np.min(loss_values):.4f}\n"
                report += f"  - Max: {np.max(loss_values):.4f}\n"
                report += f"  - Std: {np.std(loss_values):.4f}\n\n"
        
        # 保存报告
        report_path = os.path.join(self.current_dir, f'training_report_epoch_{epoch}.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"训练报告已保存: {report_path}")
        return report

def visualize_batch(images, title=None, save_path=None):
    """可视化一批图像"""
    # 创建图像网格
    grid = make_grid(images, nrow=int(np.sqrt(len(images))), normalize=True)
    
    # 转换为numpy数组并调整通道顺序
    grid = grid.cpu().numpy().transpose(1, 2, 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    if title:
        plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """绘制混淆矩阵"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    # 使用示例
    visualizer = TrainingVisualizer()
    
    # 模拟一些训练数据
    for epoch in range(5):
        # 模拟损失值
        losses = {
            'total_loss': np.random.randn(100).cumsum(),
            'gan_loss': np.random.randn(100).cumsum(),
            'recon_loss': np.random.randn(100).cumsum(),
            'perceptual_loss': np.random.randn(100).cumsum(),
            'style_loss': np.random.randn(100).cumsum(),
            'd_loss': np.random.randn(100).cumsum()
        }
        
        # 更新并绘制损失
        visualizer.update_losses(losses)
        visualizer.plot_losses(epoch)
        visualizer.plot_separate_losses(epoch)
        
        # 生成训练报告
        model_info = {
            'Generator Parameters': 1000000,
            'Discriminator Parameters': 500000,
            'Learning Rate': 0.0002,
            'Batch Size': 32
        }
        visualizer.generate_training_report(epoch, model_info) 