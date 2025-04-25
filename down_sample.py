import pclpy
from pclpy import pcl
import numpy as np

def downsample_point_cloud(input_file: str, output_file: str, leaf_size: float = 0.01):
    # Load the point cloud
    cloud = pcl.PointCloud.PointXYZRGBA()
    if pcl.io.loadPCDFile(input_file, cloud) == -1:
        print("Error loading point cloud file!")
        return
    
    print(f"Loaded point cloud with {cloud.size()} points.")
    
    # Apply Voxel Grid filtering
    voxel_filter = pcl.filters.VoxelGrid.PointXYZRGBA()
    voxel_filter.setInputCloud(cloud)
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size)
    
    downsampled_cloud = pcl.PointCloud.PointXYZRGBA()
    voxel_filter.filter(downsampled_cloud)
    
    print(f"Downsampled point cloud has {downsampled_cloud.size()} points.")
    
    # Save the downsampled point cloud
    pcl.io.save(output_file, downsampled_cloud)
    print(f"Downsampled point cloud saved to {output_file}")

def convert_pcd_format(input_file: str, output_file: str):
    cloud_rgba = pcl.PointCloud.PointXYZRGBA()
    if pcl.io.loadPCDFile(input_file, cloud_rgba) == -1:
        print("Error loading point cloud file!")
        return
    
    print(f"Loaded point cloud with {cloud_rgba.size()} points.")
    
 
    # Extract XYZ coordinates
    xyz = np.array(cloud_rgba.xyz, copy=True)
    
    # Extract RGBA values and compute intensity
    r = np.array(cloud_rgba.r, copy=True)
    g = np.array(cloud_rgba.g, copy=True)
    b = np.array(cloud_rgba.b, copy=True)
    intensity = 0.299*r + 0.587*g + 0.114*b 
    intensity = np.reshape(intensity,(len(intensity),1))
    
    # Assign values to the new point cloud
    print(np.shape(xyz),np.shape(intensity))
    cloud_array = np.append(xyz,intensity,axis = 1)
    print(np.shape(cloud_array))

    cloud_out = pcl.PointCloud.PointXYZ(cloud_array)
    
    print(f"Converted point cloud has {cloud_out.size()} points.")
    
    # Save the converted point cloud
    pcl.io.save(output_file, cloud_out)
    print(f"Converted point cloud saved to {output_file}")

def downsample_point_cloud_random(input_file: str, output_file: str, sample_ratio: float = 0.05):
    # Load the point cloud
    cloud = pcl.PointCloud.PointXYZRGBA()
    if pcl.io.loadPCDFile(input_file, cloud) == -1:
        print("Error loading point cloud file!")
        return
    
    print(f"Loaded point cloud with {cloud.size()} points.")
    
    # Apply Random Sampling
    random_filter = pcl.filters.RandomSample.PointXYZRGBA()
    random_filter.setInputCloud(cloud)
    random_filter.setSample(int(cloud.size() * sample_ratio))
    
    downsampled_cloud = pcl.PointCloud.PointXYZRGBA()
    random_filter.filter(downsampled_cloud)
    
    print(f"Downsampled point cloud has {downsampled_cloud.size()} points.")
    
    # Save the downsampled point cloud
    pcl.io.save(output_file, downsampled_cloud)
    print(f"Downsampled point cloud saved to {output_file}")

# convert_pcd_format("result/outdoor_lap/downsampled.pcd", "result/outdoor_lap/downsampled_xyzi.pcd")

downsample_point_cloud_random("result/outdoor_lap/result.pcd", "result/outdoor_lap/downsampled.pcd", sample_ratio=0.05)
