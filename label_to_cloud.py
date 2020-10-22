import numpy as np
import seaborn as sns
import mayavi.mlab as mlab

mapping = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4,
           20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11,
           49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16,
           72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6,
           255: 8, 256: 5, 257: 5, 258: 4, 259: 5}

colors = np.array(sns.color_palette('hls', 34)) * 255
colors = np.concatenate([colors.astype(np.int), np.ones([colors.shape[0], 1]) * 255], axis=1)
file_id = '000100'

if __name__ == '__main__':

  # load point clouds
  scan_dir = f'examples\\semantic_kitti\\velodyne\\{file_id}.bin'
  scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)

  # load labels
  label = np.fromfile(f'examples\\semantic_kitti\\labels\\{file_id}.label', dtype=np.int32).reshape((-1))
  label = label & 0xFFFF  # semantic label in lower half
  label = np.array([mapping[l] for l in label])

  fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))

  plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], label - 1, mode="point", figure=fig)

  # magic to modify lookup table
  plot.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, colors.shape[0])
  plot.module_manager.scalar_lut_manager.lut.number_of_colors = colors.shape[0]
  plot.module_manager.scalar_lut_manager.lut.table = colors

  mlab.view(azimuth=230, distance=50)
  mlab.savefig(filename='examples/semantic_kitti_label_to_cloud.png')
  mlab.show()
