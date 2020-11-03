import numpy as np
import seaborn as sns
import mayavi.mlab as mlab

from skimage import io

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

  # transform the pointcloud from velodyne coordiante to camera_0 coordinate
  scan_hom = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1), dtype=np.float32)))  # [N, 4]
  scan_C0 = np.dot(scan_hom, np.dot(V2C.T, R0.T))  # [N, 3]

  # transform the pointcloud from camera_0 coordinate to camera_2 coordinate
  scan_C0_hom = np.hstack((scan_C0, np.ones((scan.shape[0], 1), dtype=np.float32)))  # [N, 4]
  scan_C2 = np.dot(scan_C0_hom, P2.T)  # [N, 3]
  scan_C2_depth = scan_C2[:, 2]
  scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T

  # remove points outside the image
  inds = scan_C2_depth > 0
  inds = np.logical_and(inds, scan_C2[:, 0] > 0)
  inds = np.logical_and(inds, scan_C2[:, 0] < img.shape[1])
  inds = np.logical_and(inds, scan_C2[:, 1] > 0)
  inds = np.logical_and(inds, scan_C2[:, 1] < img.shape[0])

  scan_C2 = scan_C2[inds]
  scan_in_img = scan[inds]
  scan_not_in_img = scan[np.logical_not(inds)]

  colors = img[scan_C2[:, 1].astype(np.int), scan_C2[:, 0].astype(np.int)]
  colors = np.concatenate([colors, np.ones([colors.shape[0], 1]) * 255], axis=1)

  fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))

  # draw point cloud not in image
  # mlab.points3d(scan_not_in_img[:, 0],
  #               scan_not_in_img[:, 1],
  #               scan_not_in_img[:, 2], mode="point", figure=fig)

  # draw point cloud not in image
  plot = mlab.points3d(scan_in_img[:, 0],
                       scan_in_img[:, 1],
                       scan_in_img[:, 2],
                       np.arange(len(scan_in_img)), mode="point", figure=fig)

  # magic to modify lookup table
  plot.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, colors.shape[0])
  plot.module_manager.scalar_lut_manager.lut.number_of_colors = colors.shape[0]
  plot.module_manager.scalar_lut_manager.lut.table = colors

  mlab.view(azimuth=230, distance=50, elevation=60, focalpoint=np.mean(scan_in_img, axis=0)[:-1])
  mlab.savefig(filename='examples/kitti_img_to_cloud.png')
  mlab.show()
