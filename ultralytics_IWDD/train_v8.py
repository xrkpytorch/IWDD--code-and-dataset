import sys
import argparse
import os

sys.path.append('/root/ultralyticsPro/') # Path 以Autodl为例

from ultralytics import YOLO


def main(opt):

    yaml = opt.cfg
    model = YOLO(yaml)

    # 配置优化器
    if opt.optimizer == 'SGD':
        optimizer = 'SGD'
    elif opt.optimizer == 'Adam':
        optimizer = 'Adam'
    elif opt.optimizer == 'AdamW':
        optimizer = 'AdamW'
    else:
        raise ValueError(f"Unsupported optimizer: {opt.optimizer}")

    # 训练模型
    results = model.train(
        data='my_data/my_data.yaml',
        epochs=200,
        imgsz=640,
        workers=8,
        batch=32,
        optimizer=optimizer,
        lr0=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="ultralytics/cfg/models/cfg2024/yolov8.yaml", help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.937, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
