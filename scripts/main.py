#!/usr/bin/env python3

# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from spwc import sscweb

from space.models.planetary import Magnetosheath


from space.models.planetary import Magnetosheath

if __name__ == '__main__':
    ssc = sscweb.SscWeb()

    #    obs_list = ssc.get_observatories()

    df = ssc.get_orbit(product="mms1",
                       start_time="2020-9-20",
                       stop_time="2020-10-24").to_dataframe()

    df = df.iloc[::10]
    print(df.columns, df.size)
    # df = pd.DataFrame(np.random.rand(10000, 3) * 2 - 1, columns=list('XYZ'))

    # df['r'] = np.sqrt(np.square(df.X) + np.square(df.Y) + np.square(df.Z))
    # df['t'] = np.arctan(df.Y / df.X)
    # df['p'] = np.arctan(df.Z / df.r)

    slice1 = df  # [df.Z >= 0]
    # slice2 = df[df.r < 0.8][df.Z >= 0]
    #
    # slice1 = df[df.r < 0.2][df.Z < 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-60000, 60000)
    ax.set_ylim3d(-60000, 60000)
    ax.set_zlim3d(-60000, 60000)
    # ax.axis("equal")

    # ax.scatter(slice1.X, slice1.Y, slice1.Z, color='g')
    ax.plot(slice1.X, slice1.Y, slice1.Z, color='g')
    #    ax.scatter(slice2.X, slice2.Y, slice2.Z, color='r')

    # additional 2d plot
    #    fig2 = plt.figure()
    #
    #    filt = df[(df.Z >= 0.5) & (df.Z < 1)]
    #
    #    fig2.scatter(filt.X, filt.Z)

    plt.show()
