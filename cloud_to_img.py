import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from skimage import io
from matplotlib.lines import Line2D

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
file_id = '000010'

if __name__ == '__main__':
  # load point clouds
  scan_dir = f'examples\\kitti\\velodyne\\{file_id}.bin'
  scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)

  # load image
  img = np.array(io.imread(f'examples\\kitti\\image_2\\{file_id}.png'), dtype=np.int32)

  # load labels
  with open(f'examples\\kitti\\label_2\\{file_id}.txt', 'r') as f:
    labels = f.readlines()

  # load calibration file
  with open(f'examples\\kitti\\calib\\{file_id}.txt', 'r') as f:
    lines = f.readlines()
    P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    R0 = np.array(lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
    V2C = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

  fig = plt.figure(figsize=(12, 6))
  # draw image
  plt.imshow(img)

  # transform the pointcloud from velodyne coordiante to camera_0 coordinate
  scan_hom = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1), dtype=np.float32))) # [N, 4]
  scan_C0 = np.dot(scan_hom, np.dot(V2C.T, R0.T)) # [N, 3]

  # transform the pointcloud from camera_0 coordinate to camera_2 coordinate
  scan_C0_hom = np.hstack((scan_C0, np.ones((scan.shape[0], 1), dtype=np.float32))) # [N, 4]
  scan_C2 = np.dot(scan_C0_hom, P2.T) # [N, 3]
  scan_C2_depth = scan_C2[:, 2]
  scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T

  # remove points outside the image
  inds = scan_C2[:, 0] > 0
  inds = np.logical_and(inds, scan_C2[:, 0] < img.shape[1])
  inds = np.logical_and(inds, scan_C2[:, 1] > 0)
  inds = np.logical_and(inds, scan_C2[:, 1] < img.shape[0])
  inds = np.logical_and(inds, scan_C2_depth > 0)

  plt.scatter(scan_C2[inds, 0], scan_C2[inds, 1], c=-scan_C2_depth[inds], alpha=0.5, s=1, cmap='viridis')

  # fig.patch.set_visible(False)
  plt.axis('off')
  plt.tight_layout()
  plt.savefig('examples/kitti_cloud_to_img.png', bbox_inches='tight')
  plt.show()
