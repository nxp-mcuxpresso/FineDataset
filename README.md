# FineDataset

#### 介绍 (Introduction)

finedataset工具用于根据约束条件从现有的物体检测(OD)数据集生成新数据集，使得新生成的数据集满足特定长宽比和大小，外接矩形框的大小、形状和数量，只包含需要的类别等，从而更容易从零训练轻量级模型。

The FineDataset tool is used to generate new datasets from existing object detection datasets based on constraints that can be much easier to train lightweight models from stratch, so that the newly generated datasets meet specific aspect ratios and sizes of images; the size, shape and number of external rectangular boxes; and can only contain the required categories, etc.

## 主要功能 (main features)

- 读取多种格式的数据集 (Read multiple formats of datasets)
  
  - (zip only) **COCO-大类** (COCO-Coarse)
  
  - (zip only) **COCO-小类** (COCO-Fine)
  
  - (zip only) **Crowd Human**
  
  - (tree only) 用于一些行人检测数据集的**Seq-Vbb**   (Seq-Vbb for some pedestrian detection datasets)
  
  - (tree only) **Sub dataset **
    
    - Note: "Sub dataset" is the output format of this tool
  
  - (tar, tree) **VOC**
  
  - (zip, tree) **Wider Face**
  
  - (tree only) **YOLO**
  
  - 

- 可选择数据集的子集 (Option to select a subset of a dataset)
  
  - train
  
  - val
  
  - any 
  
  - 

- 根据高宽比筛选输入边框 (Filter input bounding boxes by min,max H:W of bounding box)

- 根据物体边框个数筛选输入图像(Filter input images by min,max bounding boxes per image)

- 从原始图像中根据可配置的约束切割子块 (Cut sub patches from input images based on configurable constraints)

- 从源数据集中筛选出类别的子集 (Select a subset of classes from original dataset)

- 生成每张图像只包含一个框的数据集 (Generate dataset that has only one bounding box per image )

- 生成每张图像包含多个框的数据集 (Generate dataset that has multiple bounding boxes per image)

- 浏览源数据集和新生成的数据集中的样本 (Browse the samples in original dataset and genreated dataset)
  
  - VOC (tar)
  
  - YOLO (tree)

- 输入数据集、约束条件可以保存成配置文件并再次加载 (The configuration of input dataset, constgraints can be saved to file and loaded later)

- 附带一个锚框配置与匹配率分析工具用于SSD家族的OD模型 (Provided an anchor box configuration tool for SSD family models) 

#### 软件架构

软件架构说明

#### 安装教程 (How to install)

1. 安装Python 3.7或更高版本，推荐使用venv或Anaconda (Install Python 3.7 or higher, recommend to use venv or Anaconda)
2. 安装依赖(Install dependencies)："python -m pip install -r requirements.txt" 
3. 运行(Run) : "python app_main.py"

#### 使用说明

1. 您需要在硬盘上先下载用于目标检测(OD)数据集 (You need to download datasets for Object Detection (OD))
2. VOC数据
3. xxxx

#### 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request

#### 特技

1. 使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2. Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3. 你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4. [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5. Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6. Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
