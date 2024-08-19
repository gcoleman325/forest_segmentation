import numpy as np
import open3d as o3d
import os
import glob

def preprocess_point_cloud(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh

def cov_metrics(pcd):
    covs = pcd.estimate_covariances()
    covs = np.asarray(pcd.covariances)

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

inFile = "D:/scans/New folder"
outFile = "D:/scans/New folder"

files = glob.glob(os.path.join(inFile, '*.pcd'))

for file_path in files:
    pcd = o3d.io.read_point_cloud(file_path)

    pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, 0.5)
    fpfh_features = np.asarray(pcd_fpfh.data).T

    points = np.asarray(pcd_down.points)
    points = points.reshape(-1, points.shape[-1])
    
    cov_metrics_array = cov_metrics(pcd_down)

    if points.shape[0] != fpfh_features.shape[0]:
        raise ValueError("Mismatch in number of points and FPFH features.")
    if points.shape[0] != cov_metrics_array.shape[0]:
        print(points.shape[0])
        print(cov_metrics_array.shape[0])
        raise ValueError("Mismatch in number of points and covariance metrics.")
    
    fpfh_with_points = np.hstack((points, fpfh_features))
    cov_with_points = np.hstack((points, cov_metrics_array))
    
    base_name = os.path.basename(file_path).replace('.pcd', '')
    output_csv1 = os.path.join(outFile, base_name + "_fpfh.csv")
    output_csv2 = os.path.join(outFile, base_name + "_cov.csv")
    
    np.savetxt(output_csv1, fpfh_with_points, delimiter=',', header='x,y,z,' + ','.join([f'feature{i}' for i in range(fpfh_features.shape[1])]), comments='')
    
    cov_headers = ['linearity', 'planarity', 'scattering', 'omnivariance', 'anisotropy', 'eigentropy', 'curvature']
    header2 = 'x,y,z,' + ','.join(cov_headers)
    np.savetxt(output_csv2, cov_with_points, delimiter=',', header=header2, comments='')