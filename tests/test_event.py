from io import StringIO

import numpy as np
import pandas as pd
import pytest

from EventTime.event import TrackData, _consecutive, _distance

df_excavator = pd.DataFrame({
    'frame': range(4),
    'object': [1] * 4,
    'left': [0] * 4,
    'top': [0] * 4,
    'width': [10, 12, 14, 16],
    'height': [10, 12, 14, 16],
    'class': [4] * 4  # excavator
})
df_truck = pd.DataFrame({
    'frame': range(4),
    'object': [2] * 4,
    'left': [10, 5, 0, 5],
    'top': [0, 0, 0, 0],
    'width': [10] * 4,
    'height': [10] * 4,
    'class': [5] * 4  # truck
})

df = pd.concat([df_excavator, df_truck])
td = TrackData(df=df, fps=1.0, classes=None)
sqrt2 = np.sqrt(2)


def test_distance():
    x = np.array([0, 0])
    y = np.array([[1, 1], [2, 2]])
    assert _distance(x=x, y=y, p=2) == pytest.approx([sqrt2, 2 * sqrt2])


def test_consecutive():
    a1 = [0, 1, 2]
    a2 = [11, 12, 13, 14]
    a3 = [41, 42, 43]

    array = np.array(a1 + a2 + a3)
    ls = _consecutive(array=array, stepsize=1, index=False)

    assert ls[0] == pytest.approx(a1)
    assert ls[1] == pytest.approx(a2)
    assert ls[2] == pytest.approx(a3)

    lsi = _consecutive(array=array, stepsize=1, index=True)
    assert lsi[0] == pytest.approx([0, 1, 2])
    assert lsi[1] == pytest.approx([3, 4, 5, 6])
    assert lsi[2] == pytest.approx([7, 8, 9])


def test_distance_by_time():
    x = np.array([[[0, 0]], [[1, 1]]])
    y = np.array([
        [[0, 0], [1, 0], [0, 2]],
        [[1, 2], [4, 1], [np.nan, 2]],
    ])

    dist = np.array([
        [0, 1, 2],
        [1, 3, np.nan],
    ])
    assert np.allclose(_distance(x=x, y=y, p=2, axis=2), dist, equal_nan=True)


def test_distance_TrackData():
    txt = '''frame class object cx cy diag left top width height
    1 1 1 0 0 1 0 0 0 0
    1 1 2 5 5 1 0 0 0 0
    1 2 3 1 0 1 0 0 0 0
    1 2 4 4 5 1 0 0 0 0
    2 1 1 1 1 1 0 0 0 0
    2 1 2 5 5 1 0 0 0 0
    2 2 3 NA 2 1 0 0 0 0
    2 2 4 1 2 1 0 0 0 0
    3 1 1 1 NA 1 0 0 0 0
    3 1 2 1 NA 1 0 0 0 0
    3 2 3 NA 1 1 0 0 0 0'''
    df_ = pd.read_csv(StringIO(txt), sep=' ', na_values='NA')
    td_ = TrackData(df=df_, fps=1.0, classes=('dummy', 'test1', 'test2'))
    td_.closest_pairs(class1=1, class2=2)

    assert 'Pair-test1' in td_.df.columns
    assert 'Pair-test2' in td_.df.columns

    assert np.allclose(td_.df['Pair-test1'],
                       np.array([
                           [3, 4, np.nan, np.nan],
                           [4, np.nan, np.nan, np.nan],
                           [np.nan, np.nan, np.nan, np.nan],
                       ]).ravel()[:-1],
                       equal_nan=True)
    assert np.allclose(td_.df['Pair-test2'],
                       np.array([
                           [np.nan, np.nan, 1, 2],
                           [np.nan, np.nan, np.nan, 1],
                           [np.nan, np.nan, np.nan, np.nan],
                       ]).ravel()[:-1],
                       equal_nan=True)


def test_get_objects():
    assert td.get_objects(cls=4) == [1]
    assert td.get_objects(cls=5) == [2]


def test_get_class():
    assert td.get_class(1) == 4


def test_trace():
    cols = df_excavator.columns
    assert td.trace(obj=1).df[cols].equals(df_excavator)
    assert td.trace(obj=2).df[cols].equals(df_truck)


if __name__ == '__main__':
    test_distance_TrackData()
