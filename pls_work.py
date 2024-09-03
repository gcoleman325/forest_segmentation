import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import KMeans
import random
import os
import glob
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import KMeans

def preprocess_point_cloud(pcd):
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    radius_normal = 0.2
    print("estimating normals")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = 0.5
    print("computing fpfh")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh = np.asarray(pcd_fpfh.data).T
    return pcd, fpfh

def cov_metrics(pcd):
    print("estimating covariances")
    covs = pcd.estimate_covariances()
    covs = np.asarray(pcd.covariances)

    print("computing eigen-based metrics")
    metrics = []
    for pt in covs:
        eigenvalues = np.linalg.eigvals(pt)
        e1, e2, e3 = eigenvalues
        
        linearity = (e1 - e2) / e1
        planarity = (e2 - e3) / e1
        scattering = e3 / e1
        omnivariance = (e1 * e2 * e3) ** (1 / 3)
        anisotropy = (e1 - e3) / e1
        eigentropy = -(e1 * np.log(e1) + e2 * np.log(e2) + e3 * np.log(e3))
        curvature = e3 / (e1 + e2 + e3)

        metrics.append((linearity, planarity, scattering, omnivariance, anisotropy, eigentropy, curvature))

    dtype = [('linearity', 'f8'), ('planarity', 'f8'), ('scattering', 'f8'), 
            ('omnivariance', 'f8'), ('anisotropy', 'f8'), ('eigentropy', 'f8'), 
            ('curvature', 'f8')]
    
    metrics_array = np.array(metrics, dtype=dtype)
  
    return np.array([tuple(row) for row in metrics_array])

## TESTING ##
# loading/preprocessing 
print("reading point cloud")
pcd = o3d.io.read_point_cloud("C:/Users/ellie/OneDrive/Desktop/lidar_local/ccb-3_preprocessed.pcd")
print("preprocessing point cloud")
pcd, fpfh = preprocess_point_cloud(pcd)
print("getting metrics")
cov = cov_metrics(pcd)

cov_headers = ['linearity', 'planarity', 'scattering', 'omnivariance', 'anisotropy', 'eigentropy', 'curvature']
header = 'x,y,z,' + ','.join([f'feature{i}' for i in range(fpfh.shape[0])]) + ','.join(cov_headers)
all_metrics = np.hstack([np.asarray(pcd.points), fpfh, cov])

print(all_metrics)