import dlib
import numpy as np

def getFiducialPoints(image, shape_predictor):
        """ This function helps get the 9 required fiducial points of an image.
        The way to get the fiducial points is as described in the PyImageSearch blog
        https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        :param image:
        :type image:
        :param shape_predictor:
        :type shape_predictor:
        :return points:
        :rtype points:
        """

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)

        rects = detector(image, 1)

        if len(rects) > 1:
                print(
                        f'Warning! There is more than 1 face in the image. Expected 1, got {len(rects)}. Continuing with one of the faces...')
        # assert len(rects) == 1, f"There is more than 1 face in the image. Expected 1, got {len(rects)}"

        shape = predictor(image, rects[0])

        # Fiducial points index order -> right eye right, right eye left, left eye right,
        # left eye left, nose right, nose tip, nose left, mouth right tip, mouth left tip
        # these indices are specific to the shape predictor being used
        idxs = [37,
                40,
                43,
                46,
                32,
                34,  # nose tip
                36,
                49,
                55]

        points = np.zeros((2, 9))

        for i, idx in enumerate(idxs):
                points[0][i] = shape.part(idx - 1).x
                points[1][i] = shape.part(idx - 1).y

        return points