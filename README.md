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
  
  - (tree only) **Sub dataset**
    
    - Note: "Sub dataset" is the output format of this tool
  
  - (tar, tree) **VOC**
  
  - (zip, tree) **Wider Face**
  
  - (tree only) **YOLO**
  
  - 

- 可选择数据集的子集 (Option to select a subset of a dataset)
  
  - train
  
  - val
  
  - any 

- 可以配置高宽比的约束筛选输入边框 (Filter input bounding boxes by min,max H:W of configurable bounding box constraints)
  
  - 当图片中包含了不符合约束条件的"脏"外接框时，可以跳过这整张图片 (Can even skip a whole image file that contains “dirty" bounding boxes that do not satisfy constraints)
  
  - 也可以设置为仍然读取图片，但是不使用不符合条件的边框 (Or configure to still use the "dirty" image but do not use the "dirty" bounding boxes)
  
  - 也可以忽略约束条件 (Or just simply ignore the constaints)

- 根据物体边框个数筛选输入图像(Filter input images by min,max bounding boxes per image)

- 从原始图像中根据可配置的约束切割子块 (Cut sub patches from input images based on configurable constraints)

- 

- 从源数据集中筛选出类别的子集 (Select a subset of classes from original dataset)

- 生成每张图像只包含一个框的数据集 (Generate dataset that has only one bounding box per image )

- 生成每张图像包含多个框的数据集 (Generate dataset that has multiple bounding boxes per image)

- 浏览源数据集和新生成的数据集中的样本 (Browse the samples in original dataset and genreated dataset)
  
  - VOC (tar)
  
  - YOLO (tree)

- 可配置新生成的数据集中全部物体面积的占比范围 (Can configure the range of the ratio of the area of all bounding boxes )

- 可配置单个物体最小面积比例(Can configure the min area ratio of a bounding boxes)

- 生成的图片中有随机设置的各方向边距(Generated images contain random margins for 4 directions)

- 输入数据集、约束条件可以保存成配置文件并再次加载 (The configuration of input dataset, constgraints can be saved to file and loaded later)

- 附带一个锚框配置与匹配率分析工具用于SSD家族的OD模型 (Provided an anchor box configuration tool for SSD family models) 

#### 软件架构 (Software architecture)

本工具使用PyQT5编写。对源数据集和导出数据集格式的支持都通过插件系统完成。在生成子块数据集时，使用了手动编写的子块聚类规则。 (This tool is written with PyQT5, and uses plugin-oriented design for reading source dataset and exporting dataset formats. When generating sub-patch datasets, it uses manual-crafted rules for clustering)

#### 安装教程 (How to install)

- [ ] 安装Python 3.7或更高版本，推荐使用venv或Anaconda (Install Python 3.7 or higher, recommend to use venv or Anaconda)

- [ ] 安装依赖(Install dependencies)："python -m pip install -r requirements.txt" 

- [ ] 运行(Run): python ./app_main.py

#### 源数据集的准备 (Prepare source datasets)

在使用前，您需要在硬盘上先下载用于目标检测(OD)数据集 (You need to download datasets for Object Detection (OD))

1. 对于VOC, COCO, Wider face数据集，建议**不要**解压缩 (For VOC, COCO, Wider face datasets, we suggest to keep the archives, no need to extract.)
2. 对于 Seq-Vbb 格式的数据集，需要在数据集目录下建立两个子目录，分别是"annotations"和"images"。这里面也可以再有子目录结构，但是必须保证"annotations/.../xyz.vbb" 文件有对应的"images/.../xyz.seq"  (For Seq-Vbb datasets, you need to create two subfolders in dataset folder, one is "annotations", the other is "images". You can further add nested subfolders under them, as long as each "annotations/.../xyz.vbb" is paired with "images/.../xyz.seq")
3. 对于 CrowdHuman数据集，需要在数据集目录下放置CrowdHuman数据集的压缩包和标注文件（For CrowdHuman dataset, put related archives and label files under dataset's folder)： 
   1. "CrowdHuman_train##.zip", "CrowdHuman_val.zip", "annotation_train.odgt", "annotation_val.zip".

1. 在本仓库的主目录下，运行(In the root folder of this repo, Run) : 
   
   "python app_main.py"

#### 生成子块数据集 (Generate sub-patch datasets)

通过“生成子块数据集"中的”单框"和"多框"按钮生成新数据集。在生成期间可以点击"中止"以停止，此时会留下已经生成的部分。生成的数据集存储在 (To generate new sub datasets, click "1-box" or "N-box" buttons under "Generate sub-patch dataset" frame, and you can click "Abort" during generating process, this will leave partial generated contents. The genreated dataset is located in) 

<root>/outs/out_{any|train|val}_{single|multi}[_YYYY-MM-dd-hh-mm-ss]"

1. 只有在勾选了“输出目录有时间戳”时才会追加 YYYY-MM-dd-hh-mm-ss 后缀，否则，再次生成相同目录时会**删除上一次**生成的结果。(The "YYYY-MM-dd-hh-mm-ss" postfix is only added when the "Timestamp in out folder" is checked, otherwise, when generating sub datasets, it will DELETE previous generated folder first!)

生成的数据集是本工具自定义的一种简单的格式，名为"Sub dataset", 在同一个目录下存储图片和json格式的标注文件。生成的数据集也可以读取进来做为源数据集来进一步生成子块数据集 (Generated dataset is in a custom simple format defined by this tool, named as "Sub dataset", it stores images and json labels in the same folder. Generated dataset can also be loaded as source dataset to generate sub-patch datasets)。

1. 如果要读取生成的数据集，强烈建议先把它复制到其它位置 (If you'd like to load previous generated sub-patch dataset, we strongly recommend you move it elsewhere before you operate)。

#### 导出数据集 (Export dataset)

在"导出数据集格式"中选择要导出的格式，再点击"导出"按钮。本工具会在"outs"目录下扫描全部已生成的数据集，执行导出操作，并存储为<export_format>_<folder_name>.tar。例如，"voc_out_any.tar" (Select export format in "Export format" dropbox, then click "Export" button, this tool then scan all generated datasets under "outs" folder, do export, and save the exported dataset as "<export_format>_<folder_name>.tar")

#### 参与贡献 (Contribution)

1. Fork
2. Branch Feat_xxx
3. Submit
4. Pull Request
