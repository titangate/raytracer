import numpy


def rotate(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = numpy.asarray(axis)
    theta = numpy.asarray(theta)
    axis = axis/numpy.sqrt(numpy.dot(axis, axis))
    a = numpy.cos(theta/2)
    b, c, d = -axis*numpy.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return numpy.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0],
                       [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0],
                       [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0],
                       [0, 0, 0, 1]])


def scale(x, y=None, z=None):
    if y is None:
        y = x
        z = x

    return numpy.array([[x, 0, 0, 0],
                       [0, y, 0, 0],
                       [0, 0, z, 0],
                       [0, 0, 0, 1]])


def translate(x, y, z):
    return numpy.array([[1, 0, 0, x],
                       [0, 1, 0, y],
                       [0, 0, 1, z],
                       [0, 0, 0, 1]])


def apply(mat, vec):
    return mat.dot(numpy.concatenate((vec,[1.])))[:3]
