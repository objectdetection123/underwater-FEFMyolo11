from ultralytics import YOLO

if __name__ == "__main__":
    model_path ='runs/detect/train305/weights/best.pt'
    data_path = 'ultralytics/cfg/datasets/underwater3.yaml'
    # 加载模型
    model = YOLO(model_path)
    # 验证模型
    metrics = model.val(
        val=True,  # (bool) 在训练期间进行验证/测试
        data=data_path,
        split="test",  # (str) 用于验证的数据集拆分，例如'val'、'test'或''
        batch=32,  # (int) 每批的图像数量（-1 为自动批处理）
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        device='0',  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        #workers=8,  # 数据加载的工作线程数（每个DDP进程）
        #save_json=False,  # 保存结果到JSON文件
        #save_hybrid=False,  # 保存标签的混合版本（标签 + 额外的预测）
        conf=0.08,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
        iou=0.83,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        project="runs/test",  # 项目名称（可选）
        name="exp",  # 实验名称，结果保存在'project/name'目录下（可选）
       # max_det=300,  # 每张图像的最大检测数
        half=True,  # 使用半精度 (FP16)
        #dnn=False,  # 使用OpenCV DNN进行ONNX推断
        plots=True,  # 在训练/验证期间保存图像
    )

    print(f"mAP50-95: {metrics.box.map}")  # map50-95
    print(f"mAP50: {metrics.box.map50}")  # map50
    print(f"mAP75: {metrics.box.map75}")  # map75
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}")  # FPS