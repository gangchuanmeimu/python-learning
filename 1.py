import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image
from modelscope.msdatasets import MsDataset
import warnings
warnings.filterwarnings('ignore')

# ====================== 全局配置（适配1个样本+CPU） ======================
if __name__ == '__main__':
    # 1. 基础配置
    DEVICE = torch.device('cpu')
    BATCH_SIZE = 1  # 仅1个样本，固定为1
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 10  # 减少训练轮数，适配示例样本
    IMAGE_SIZE = (224, 224)
    BEST_ACC = 0.0

    # 2. 人体属性定义
    ATTR_CLASSES = [
        'gender', 'age', 'hat', 'glasses', 'mask', 'upper_clothes',
        'lower_clothes', 'shoes', 'bag', 'umbrella', 'handbag', 'backpack',
        'cap', 'scarf', 'gloves', 'watch', 'ring', 'necklace', 'earrings'
    ]
    NUM_ATTRS = len(ATTR_CLASSES)

    # ====================== 1. 数据集加载与预处理（修复图片加载） ======================
    # 加载数据集
    print("📥 加载数据集...")
    ds = MsDataset.load(
        'DatatangBeijing/208914BoundingBoxes_HumanBodyAttributesDataInSurveillanceScenes',
        subset_name='default',
        split='train',
        cache_dir='./modelscope_cache'
    )
    print(f"📊 数据集样本数: {len(ds)}")

    # 简化数据增强（适配1个样本）
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 自定义数据集类（修复图片加载逻辑）
    class HumanAttrDataset(Dataset):
        def __init__(self, ms_dataset, transform=None):
            self.dataset = ms_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            try:
                sample = self.dataset[idx]
                
                # 修复：确保获取的是图片路径而非文件对象
                img_path = sample['image']
                if isinstance(img_path, str):
                    # 路径是字符串，正常打开
                    img = Image.open(img_path).convert('RGB')
                else:
                    # 路径是文件对象，转换为路径字符串
                    img_path = img_path._path if hasattr(img_path, '_path') else str(img_path)
                    img = Image.open(img_path).convert('RGB')

                # 模拟标注（固定值，适配1个样本）
                attr_labels = torch.tensor([
                    1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0
                ], dtype=torch.float32)

                if self.transform:
                    img = self.transform(img)

                return img, attr_labels

            except Exception as e:
                print(f"❌ 样本{idx}处理错误: {e}")
                # 兜底数据
                dummy_img = torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1])
                dummy_label = torch.zeros(NUM_ATTRS, dtype=torch.float32)
                return dummy_img, dummy_label

    # 初始化数据集
    train_dataset = HumanAttrDataset(ds, transform=train_transform)
    val_dataset = HumanAttrDataset(ds, transform=val_transform)
    print("⚠️  仅检测到1个示例样本，复用数据进行训练/验证（完整数据需申请授权）")

    # 数据加载器（关闭洗牌，适配1个样本）
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    print(f"✅ 数据加载完成 | 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # ====================== 2. 模型定义（移除BatchNorm，适配1个样本） ======================
    class ResNetAttrModel(nn.Module):
        def __init__(self, num_attrs=NUM_ATTRS):
            super().__init__()
            # 加载预训练ResNet18
            self.backbone = models.resnet18(pretrained=True)
            
            # 冻结大部分参数，避免过拟合
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # 替换分类头（移除BatchNorm，适配批次=1）
            in_feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_feat, 256),
                nn.ReLU(),
                nn.Linear(256, num_attrs),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.backbone(x)

    # 初始化模型
    model = ResNetAttrModel().to(DEVICE)
    print("✅ 模型加载完成（移除BatchNorm，适配1个样本）")

    # ====================== 3. 优化器与损失函数 ======================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 简化学习率策略
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 损失函数
    criterion = nn.BCELoss()

    # ====================== 4. 训练/验证函数 ======================
    def train_one_epoch(model, loader, criterion, optimizer, epoch):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # 前向传播
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            preds = (outputs > 0.5).float()
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.numel()
            total_loss += loss.item()

            # 打印进度
            print(f"📈 Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f} | Acc: {100*correct/labels.numel():.2f}%")

        avg_loss = total_loss / len(loader)
        avg_acc = 100 * total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(model, loader, criterion):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 计算准确率
            preds = (outputs > 0.5).float()
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.numel()

        avg_loss = total_loss / len(loader)
        avg_acc = 100 * total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, avg_acc

    # ====================== 5. 主训练循环 ======================
    print("\n🚀 开始训练（CPU适配模式）")
    print("-" * 50)

    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion)
        # 学习率更新
        scheduler.step()

        # 打印结果
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS}")
        print(f"🔴 训练 | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"🟢 验证 | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"📉 学习率: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)

        # 保存最佳模型
        if val_acc > BEST_ACC:
            BEST_ACC = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': BEST_ACC,
                'attr_classes': ATTR_CLASSES
            }, 'best_human_attr_model.pth')
            print(f"✅ 保存最佳模型 | 最佳准确率: {BEST_ACC:.2f}%")

    # ====================== 训练完成 ======================
    print("\n🎉 训练完成！")
    print(f"🏆 最终最佳准确率: {BEST_ACC:.2f}%")
    print(f"💾 模型保存路径: ./best_human_attr_model.pth")

    # 验证模型加载
    checkpoint = torch.load('best_human_attr_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\n✅ 已加载最佳模型，可用于推理！")