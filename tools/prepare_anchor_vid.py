import json 
import numpy as np

vid_training_json = "/home/dc2-user/zzj/pysot/data/tracking/ed/vid/train.json"
context_amount = 0.5
exemplar_size = 127

loc_list = []
with open(vid_training_json) as f:
    data_info = json.load(f)
    for video in data_info.keys():
        for file in data_info[video].keys():
            for image in data_info[video][file].keys():
                image_anno = data_info[video][file][image]
                w, h = image_anno[2]-image_anno[0], image_anno[3]-image_anno[1]
                wc_z = w + context_amount * (w+h)
                hc_z = h + context_amount * (w+h)
                s_z = np.sqrt(wc_z * hc_z)
                scale_z = exemplar_size / s_z
                w = w*scale_z
                h = h*scale_z
                loc_list.append([w,h])

from sklearn.cluster import KMeans
data = np.array(loc_list)
estimator = KMeans(n_clusters=5)#构造聚类器
estimator.fit(data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
print(centroids)
