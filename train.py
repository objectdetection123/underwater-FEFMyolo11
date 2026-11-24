
# # 普通训练启动方式
# #model = YOLO("yolov8n.pt")
# #model.train(data="yolo-test.yaml",workers=0,batch=16,epochs=50)

# #yolo task=detect mode=train model=ultralytics/cfg/models/v8/yolov8s-spdconv.yaml data=ultralytics/cfg/datasets/VisDrone.yaml batch=4 epochs=200 imgsz=640 workers=0 cache=True resume=true device=0
# #yolo task=detect mode=train model=runs/detect/exp/weights/last.pt data=ultralytics/datasets/mydata.yaml epochs=100 save=True resume=True
# #from ultralytics import RTDETR
# #from ultralytics import YOLO
# #if __name__ == '__main__':
#     #将下面的路径换成之前训练文件中的最后的权重last.py，该文件在run文件夹下。
#     #model = YOLO(r'H:\ultralytics-main\runs\detect\train2\weights\last.pt')
#     #下面的命令不变。
#     #model.train(resume=True)  #断点续训

# #yolo task=detect mode=train model=runs/detect/train4/weights/last.pt data=ultralytics/datasets/VisDrone.yaml batch=4 epochs=200 imgsz=640 workers=0  save=True resume=True device=0
# from ultralytics import YOLO

# if __name__ == '__main__' :
#  # 加载一个模型
#   model = YOLO('runs/detect/train324/weights/last.pt')  # 从权重文件初始化模型
#   #model = YOLO('ultralytics/cfg/models/11/yolo11s.yaml')  # 从YAML建立一个新模型
#  # 训练模型
#   results = model.train(
#   data='ultralytics/cfg/datasets/underwater3.yaml',
#   device='0',
#   epochs=300,  # 训练过程中整个数据集将被迭代多少次,显卡不行你就调小点
#   batch=8,  # 一次看完多少张图片才进行权重更新

#   verbose=True,
#   patience=0,
#   resume=True,
#   save=True,
#   imgsz=640)


# 普通训练启动方式
#model = YOLO("yolov8n.pt")
#model.train(data="yolo-test.yaml",workers=0,batch=16,epochs=50)

#yolo task=detect mode=train model=ultralytics/cfg/models/v8/yolov8s-spdconv.yaml data=ultralytics/cfg/datasets/VisDrone.yaml batch=4 epochs=200 imgsz=640 workers=0 cache=True resume=true device=0
#yolo task=detect mode=train model=runs/detect/exp/weights/last.pt data=ultralytics/datasets/mydata.yaml epochs=100 save=True resume=True
#from ultralytics import RTDETR
#from ultralytics import YOLO
#if __name__ == '__main__':
    #将下面的路径换成之前训练文件中的最后的权重last.py，该文件在run文件夹下。
    #model = YOLO(r'H:\ultralytics-main\runs\detect\train2\weights\last.pt')
    #下面的命令不变。
    #model.train(resume=True)  #断点续训

#yolo task=detect mode=train model=runs/detect/train4/weights/last.pt data=ultralytics/datasets/VisDrone.yaml batch=4 epochs=200 imgsz=640 workers=0  save=True resume=True device=0
from ultralytics import YOLO

if __name__ == '__main__' :
 # 加载一个模型
  model = YOLO('runs/detect/train72/weights/last.pt')  # 从权重文件初始化模型
  #model = YOLO('ultralytics/cfg/models/11/yolo11s.yaml')  # 从YAML建立一个新模型
 # 训练模型
  results = model.train(
  data='ultralytics/cfg/datasets/underwater3.yaml',
  device='0',
  epochs=300,  # 训练过程中整个数据集将被迭代多少次,显卡不行你就调小点
  batch=8,  # 一次看完多少张图片才进行权重更新

  verbose=True,
  patience=0,
  resume=True,
  save=True,
  imgsz=640)

