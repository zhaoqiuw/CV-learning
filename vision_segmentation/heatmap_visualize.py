import cv2

import numpy as np
import os
import torch
import matplotlib.pyplot as plt

save_dir = '/heat/'


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,save_dir = '/heat/',num=0):
    i=num
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                cv2.imwrite(os.path.join(save_dir, str(i)+'.png'), superimposed_img)
                i=i+1
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmaps = cv2.resize(heatmaps, (512, 512))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                superimposed_img = cv2.resize(superimposed_img, (512,512))
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir, str(i)+'.png'), superimposed_img)
                i=i+1
#整合后的热力图代码
def featuremap_2_heatmap(feature_map, save_dir = '/home/zhaoqiu/code/mmsegmentation/attention_visualize/', name="query"):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = np.array(feature_map.cpu())
    combined_feature_map = np.sum(heatmap, axis=1)
    min_val, max_val = np.min(combined_feature_map), np.max(combined_feature_map)
    normalization_feature_map = (combined_feature_map- min_val) / (max_val - min_val)
    normalization_feature_map = np.uint8(normalization_feature_map * 255)
    normalization_feature_map = normalization_feature_map[0]
    normalization_feature_map = cv2.applyColorMap(normalization_feature_map, cv2.COLORMAP_JET)
    print(type(normalization_feature_map),normalization_feature_map.shape)
    cv2.imwrite(save_dir+name+'.png',normalization_feature_map)

