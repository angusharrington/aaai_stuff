import numpy as np
from scipy.interpolate import griddata


def UpsamplePtCloud(pc):
    """This function increases the number of points in the point cloud
    :param pc: point cloud object returned by open2d.io.read_point_cloud(filename)
    :type pc:
    :return: a numpy array containing the upsampled datapoint coordinates
    """

    # ^ follows the Sphinx docstring format

    print('Now doing the upsampling process of the point cloud')

    # get the coordinate max bounds of the point cloud - we don't worry about the z direction, hence blank
    xmax, ymax, _ = pc.get_max_bound()
    xmin, ymin, _ = pc.get_min_bound()

    # create a meshgrid of points. This will define the resolution of our upsampled point cloud
    x1 = np.linspace(xmin, xmax, 800)
    y1 = np.linspace(ymin, ymax, 1000)
    xq, yq = np.meshgrid(x1, y1)  # same values as matlab

    # print(f' xq.shape:{xq.shape} and yq.shape:{yq.shape}')
    # print(xq[0:9, 1])

    ll = np.asarray(pc.points)  # convert the data values of pc to a numpy array
    print(f'Original point cloud shape is {ll.shape}')

    # we now want to set up the coordinates of the points for the griddata interpolation
    ll_x = np.expand_dims(ll[:, 0], axis=1)  # this is just adding an extra dimension for easier concatenation
    ll_y = np.expand_dims(ll[:, 1], axis=1)
    ll_xy = np.concatenate((ll_x, ll_y), axis=1)
    ll_z = np.expand_dims(ll[:, 2], axis=1)

    # carrying out the interpolation
    vq = griddata(ll_xy, ll_z, (xq, yq), method='linear')  # interpolation method = 'linear' by default
    # print('vq shape: ',vq.shape)
    # print('vq[500, 199:209] ', vq[500,199:209]) # looks like both vqs are the same

    # set all NaN values in our griddata to zero
    where_are_NaNs = np.isnan(vq)
    vq[where_are_NaNs] = 0

    p1 = np.reshape(xq, (-1, 1), order='F')  # this reshapes (m, n) to (_, 1) where _ is automatically set

    # print('p1', p1.shape, p1[499])
    p2 = np.reshape(yq, (-1, 1), order='F')
    p3 = np.reshape(vq, (-1, 1), order='F')

    new_locations = np.concatenate((p1, p2, p3), axis=1)
    new_locations = np.concatenate((new_locations, ll), axis=0)
    print(f'Shape of new locations is {new_locations.shape}')

    # print(f'new locations 99 {new_locations[99, :]}')

    return new_locations
