import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import KMeans
import random
import os
import glob
from sklearn.naive_bayes import BernoulliNB

def preprocess_point_cloud(pcd):
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



def process_files(inFile, outFile):
    files = glob.glob(os.path.join(inFile, '*.pcd'))

    for file_path in files:
        pcd = o3d.io.read_point_cloud(file_path)

        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, 0.1)
        fpfh_features = np.asarray(pcd_fpfh.data).T

        points = np.asarray(pcd_down.points)
        points = points.reshape(-1, points.shape[-1])
        
        cov_metrics_array = cov_metrics(pcd_down)
        
        fpfh_with_points = np.hstack((points, fpfh_features))
        cov_with_points = np.hstack((points, cov_metrics_array))
        
        base_name = os.path.basename(file_path).replace('.pcd', '')
        output_csv1 = os.path.join(outFile, base_name + "_fpfh.csv")
        output_csv2 = os.path.join(outFile, base_name + "_cov.csv")
        
        np.savetxt(output_csv1, fpfh_with_points, delimiter=',', header='x,y,z,' + ','.join([f'feature{i}' for i in range(fpfh_features.shape[1])]), comments='')
        
        cov_headers = ['linearity', 'planarity', 'scattering', 'omnivariance', 'anisotropy', 'eigentropy', 'curvature']
        header2 = 'x,y,z,' + ','.join(cov_headers)
        np.savetxt(output_csv2, cov_with_points, delimiter=',', header=header2, comments='')

def likely_objects(preprocessed_path):
    pcd = o3d.io.read_point_cloud(preprocessed_path)
    pcd, pcd_fpfh = preprocess_point_cloud(pcd)
    cov = cov_metrics(pcd)
    all_metrics = np.hstack([pcd_fpfh, cov])

    pcd = np.asarray(pcd.points)
    no_trees = o3d.io.read_point_cloud(preprocessed_path.replace('preprocessed', 'no_trees'))
    no_trees, no_trees_fpfh = preprocess_point_cloud(no_trees)
    no_trees = np.asarray(no_trees.points)
    cylinder = np.ones((pcd.shape[0], 1))
    pcd = np.hstack((pcd, cylinder))
    mask = np.isin(pcd, no_trees).all(axis=1)
    pcd[mask, -1] = 0

    return pcd

def testing(self):
    df = pd.read_csv("D:/scans/preprocessed_pcd/CCB-1-1_fpfh.csv")

    n_clusters = 100
    model = KMeans(n_clusters=n_clusters)
    model.fit(df)
    predictions = model.predict(df)
    predictions = np.array(predictions)

    xyz = df.iloc[:,:3]
    xyz = [(row['x'], row['y'], row['z']) for _, row in df.iterrows()]
    xyz = np.array(xyz)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

    objects = []
    for i in range(n_clusters):
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        indices = np.where(predictions[:] == i)[0]
        object = pcd.select_by_index(indices)
        object.paint_uniform_color([R, G, B])
        objects.append(object)

    o3d.visualization.draw_geometries([objects])

pcd = o3d.io.read_point_cloud("C:/Users/ellie/OneDrive/Desktop/lidar_local/ccb-3-2-2go.pcd")
pcd = preprocess_point_cloud(pcd)
o3d.visualization.draw_geometries([pcd])