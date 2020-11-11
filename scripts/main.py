#!/usr/bin/env python3

import sys
sys.path.append("..") # for s

import datetime
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from spwc import sscweb

from space.models.planetary import Magnetosheath


class SelectionObject:

    pass


class Orbit:
    def __init__(self, orbit : np.array): # TODO maybe r_theta_phi-support?
        self.data = orbit

    def selection_intervals(self, object: SelectionObject):
        """
        Returns the ranges of the orbit selected by the object.
        """
        return object.select(self)


def _index_list_to_ranges(indices: list()):
    split_idx = np.flatnonzero(np.diff(indices, prepend=indices[0], append=indices[-1]) != 1)
    bounding_points = np.transpose([split_idx[:-1], split_idx[1:]])
    return [range(indices[n], indices[m-1]+1) for n, m in bounding_points.tolist()]


class Sphere(SelectionObject):
    def __init__(self, position: tuple, r: float):
        self.position = position
        self.radius = r

    def select(self, orbit: Orbit):
        dist = np.linalg.norm(self.position - orbit.data, axis=1)
        return list(np.where(dist <= self.radius)[0])

    def to_mesh(self):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.position[0] + self.radius * np.cos(u)*np.sin(v)
        y = self.position[1] + self.radius * np.sin(u)*np.sin(v)
        z = self.position[2] + self.radius * np.cos(v)

        return np.array((x, y, z), dtype=float)


class Cuboid(SelectionObject):
    def __init__(self, p1: tuple, p2: tuple, p3: tuple, p4: tuple):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
        self.p4 = np.array(p4)

    def select(self, orbit: Orbit):
        u = self.p1 - self.p2
        v = self.p1 - self.p3
        w = self.p1 - self.p4

        print(u,v,w)

        uP1 = np.dot(u, self.p1)
        uP2 = np.dot(u, self.p2)
        if uP1 > uP2:
            uP1, uP2 = uP2, uP1
        uO  = np.dot(orbit.data, u)

        print(uP1, uP2, uO)

        vP1 = np.dot(v, self.p1)
        vP3 = np.dot(v, self.p3)
        if vP1 > vP3:
            vP1, vP3 = vP3, vP1
        vO  = np.dot(orbit.data, v)

        print(vP1, vP3, vO)

        wP1 = np.dot(w, self.p1)
        wP4 = np.dot(w, self.p4)
        if wP1 > wP4:
            wP1, wP4 = wP4, wP1
        wO  = np.dot(orbit.data, w)

        print(wP1, wP4, wO)

        return list(np.where(
            ((uP1 <= uO) & (uO <= uP2)) &
            ((vP1 <= vO) & (vO <= vP3)) &
            ((wP1 <= wO) & (wO <= wP4))
        )[0])

    def to_mesh(self):
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        p4 = self.p4

        X = [[p1[0], p2[0], p2[0], p1[0], p1[0]],
             [p1[0], p2[0], p2[0], p1[0], p1[0]],
             [p1[0], p2[0], p2[0], p1[0], p1[0]],
             [p1[0], p2[0], p2[0], p1[0], p1[0]]]

        Y = [[p1[1], p1[1], p3[1], p3[1], p1[1]],
             [p1[1], p1[1], p3[1], p3[1], p1[1]],
             [p1[1], p1[1], p1[1], p1[1], p1[1]],
             [p3[1], p3[1], p3[1], p3[1], p3[1]]]

        Z = [[p1[2], p1[2], p1[2], p1[2], p1[2]],
             [p4[2], p4[2], p4[2], p4[2], p4[2]],
             [p1[2], p1[2], p4[2], p4[2], p1[2]],
             [p1[2], p1[2], p4[2], p4[2], p1[2]]]

        return np.array((X, Y, Z), dtype=float)

class T:
    def __init__(self, data):
        self.data = data

def test():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    cuboid = Cuboid((0, 0, 0),
                    (10, 0, 0),
                    (0, 10, 0),
                    (0, 0, 10))

    point = np.array([[-2, -2, -2],
                      [1, 1, 1],
                      [5, 5, 5],
                      [7, 7, 7],
                      [10, 10, 10],
                      [11, 11, 11]
                      ])
    t = T(point)

    sel = cuboid.select(t)
    print(sel)



    o = cuboid.to_mesh()
    ax.plot_wireframe(o[0], o[1], o[2], color="b")
    ax.scatter(point[:, 0], point[:, 1], point[:, 2], color="r")

    plt.show()



if __name__ == '__main__':
    #test()
    #sys.exit()





    ssc = sscweb.SscWeb()

    df = ssc.get_orbit(product="mms1",
                       start_time="2020-10-10",
                       stop_time="2020-10-24"
                       )
    orbit = Orbit(df.data[::2, 0:3])


    selection = []

    sphere = Sphere((30000, 30000, 30000), 15000)
    selection += orbit.selection_intervals(sphere)

    cuboid = Cuboid((10000, 10000, 10000),
                    (25000, 10000, 10000),
                    (10000, 25000, 10000),
                    (10000, 10000, 25000))

    selection += orbit.selection_intervals(cuboid)


    ranges = _index_list_to_ranges(sorted(selection))

    print('selected intervals', len(ranges))
    for r in ranges:
        print('  ',
              datetime.datetime.fromtimestamp(df.time[r.start]).strftime('%c'), '-',
              datetime.datetime.fromtimestamp(df.time[r.stop]).strftime('%c'))


    # slice1 = selected indices
    slice1 = orbit.data[selection]

    # slice2 = the other indices
    mask = np.ones(len(orbit.data), dtype=bool)
    mask[selection] = False
    slice2 = orbit.data[mask]

    # df = pd.DataFrame(np.random.rand(10000, 3) * 2 - 1, columns=list('XYZ'))

    # df['r'] = np.sqrt(np.square(df.X) + np.square(df.Y) + np.square(df.Z))
    # df['t'] = np.arctan(df.Y / df.X)
    # df['p'] = np.arctan(df.Z / df.r)

    # slice2 = df[df.r < 0.8][df.Z >= 0]
    #
    # slice1 = df[df.r < 0.2][df.Z < 0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")

    ax.set_xlim3d(5000, 32000)
    ax.set_ylim3d(5000, 32000)
    ax.set_zlim3d(5000, 32000)
    # ax.axis("equal")

    # ax.scatter(slice1.X, slice1.Y, slice1.Z, color='g')
    ax.scatter(slice1[:, 0], slice1[:, 1], slice1[:, 2], color='g')
    ax.scatter(slice2[:, 0], slice2[:, 1], slice2[:, 2], color='r')

    o = sphere.to_mesh()
    ax.plot_wireframe(o[0], o[1] ,o[2], color="b")

    o = cuboid.to_mesh()
    ax.plot_wireframe(o[0], o[1] ,o[2], color="b")

    plt.show()
