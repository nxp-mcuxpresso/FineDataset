import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

g_egCenters = [
    [5,5, 30, 42]
]

class CClusterize:
    def __init__(self, pts:np.ndarray=None):
        pass
    def ShowExample(self):
        # X为样本特征，Y为样本簇类别，共1000个样本，每个样本2个特征，对应x和y轴，共4个簇，
        # 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
        X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                        cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9)
                        
        plt.scatter(X[:, 0], X[:, 1], marker='o')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
        y_pred = KMeans(n_clusters=int(input('聚类数')), random_state=9).fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.show()

    def ShowWfUtils(self):
        import json
        import os.path as path
        strDSFolder = 'q:/datasets/wider_face/out_train_multi'
        self.gtAspects, self.gtSizes, self.gtCenters = [], [], []
        with open("%s/bboxes.json" % (strDSFolder)) as fd:
            lst = json.load(fd)
        cnt = len(lst)
        
        # 获取图片输入尺寸
        strFile = path.join(strDSFolder, path.split(lst[0]['filename'])[1])

        for (i, item) in enumerate(lst):
            for xyxy in item['xyxys']:
                cx, cy = (xyxy[0] + xyxy[2]) // 2 , (xyxy[1] + xyxy[3]) // 2
                w,h = (xyxy[2] - xyxy[0]) , (xyxy[3] - xyxy[1])
                area = w * h
                r = h / w
                self.gtAspects.append([r, 1]) # 1只是占位，为了生成2维数据
                self.gtSizes.append([area, 1])
                self.gtCenters.append([cx, cy])        

        self.gtCenters = np.array(self.gtCenters)
        self.gtSizes = np.array(self.gtSizes)
        self.gtAspects = np.array(self.gtAspects)

        
        X = self.gtAspects
        
        clusterCnt = int(input('请输入聚类数：'))

        kmObj = KMeans(n_clusters=clusterCnt, random_state=9)
        y_pred = kmObj.fit_predict(X)

        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.show()
        a = np.zeros([clusterCnt])
        b = a.copy()
        for i in range(len(X)):
            a[y_pred[i]] += X[i][0]
            b[y_pred[i]] += 1
        a /= b
        return 0
if __name__ == '__main__':



    obj = CClusterize()
    obj.ShowWfUtils()
    obj.ShowExample()