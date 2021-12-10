import sys
from itertools import product
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, Union

sys.path.insert(0, './yolov5')

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.dates import DateFormatter
from tqdm import tqdm

from EventTime.ffmpeg import FFmpeg
from yolov5.utils.plots import Annotator, colors


def _distance(x: np.ndarray, y: np.ndarray, p=2, axis=1):
    return np.sum(np.abs(y - x)**p, axis=axis)**(1.0 / p)


def _consecutive(array: np.ndarray,
                 stepsize=1,
                 index=False) -> List[np.ndarray]:
    """
    array를 stepsize만큼 증가하는 연속적인 숫자들의 덩어리로 나눔.
    [0, 12, 13, 21, 22, 23] -> [[0], [12, 13], [21, 22, 23]]

    https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array

    Parameters
    ----------
    array : np.ndarray
    stepsize : int, optional
    index : bool, optional
        `True`인 경우 원래 값 대신 index로 이루어진 array 반환

    Returns
    -------
    List[np.ndarray]

    Raises
    ------
    ValueError
        if array.ndim != 1
    """
    if array.ndim != 1:
        raise ValueError
    return np.split((np.arange(array.size) if index else array),
                    np.where(np.diff(array) != stepsize)[0] + 1)


def intensity_difference(image1: np.ndarray,
                         image2: np.ndarray,
                         maxval: Optional[float] = None,
                         inter=cv.INTER_LANCZOS4) -> float:
    """
    표준화한 두 영상의 밝기 차이 평균 산정. 두 영상의 dtype은 같다고 가정.

    Parameters
    ----------
    image1 : np.ndarray
    image2 : np.ndarray
    maxval : Optional[float], optional
        영상 밝기 최대값.
        None으로 설정하면 image1.dtype의 최대값으로 설정
    inter : int, optional
        interpolation 방법, by default cv.INTER_LANCZOS4

    Returns
    -------
    float
    """
    if image1.shape[:2] == image2.shape[:2]:
        image2r = image2
    else:
        image2r = cv.resize(image2,
                            dsize=(image1.shape[1], image1.shape[0]),
                            interpolation=inter)

    if maxval is None:
        maxval = np.iinfo(image1.dtype).max
        if not maxval != np.iinfo(image2r.dtype).max:
            raise ValueError('dtype이 일치하지 않음')

    std1 = image1.astype(float) / maxval
    std2 = image2r.astype(float) / maxval

    return np.average(np.abs(std1 - std2))


class TrackData:
    _MOT_COLS = ('frame', 'object', 'left', 'top', 'width', 'height', 'conf',
                 'x', 'y', 'z', 'class')  # MOT + class format
    _DROP_COLS = ('conf', 'x', 'y', 'z')
    _COLS = ('frame', 'time', 'class', 'object', 'left', 'top', 'width',
             'height', 'cx', 'cy', 'diag')
    _DEFAULT_CLASSES = ('StaticCrane', 'Crane', 'Roller', 'Bulldozer',
                        'Excavator', 'Truck', 'Loader', 'PumpTruck',
                        'ConcreteMixer', 'PileDriving', 'OtherVehicle')

    def __init__(self,
                 df: Union[PathLike, pd.DataFrame],
                 fps: float,
                 classes: Optional[Tuple[str]] = None) -> None:
        if not isinstance(df, pd.DataFrame):
            df = pd.read_csv(df, sep=' ', index_col=False, names=self._MOT_COLS)

        if any(x not in df.columns for x in ['cx', 'cy', 'diag']):
            df.loc[:, 'cx'] = df['left'] + df['width'] / 2.0
            df.loc[:, 'cy'] = df['top'] + df['height'] / 2.0
            df.loc[:, 'diag'] = np.sqrt(
                np.square(df['width']) + np.square(df['height']))

        if not df.index.is_unique:
            logger.warning('track 데이터에 중복된 index가 있습니다. index를 새로 설정합니다.')
            df.reset_index(drop=True, inplace=True)

        df = df[[x for x in df.columns if x not in self._DROP_COLS]]
        df['time'] = self.frame2time(frames=df['frame'].values, fps=fps)

        self._df: pd.DataFrame = df
        self._fps = fps
        self._classes = self._DEFAULT_CLASSES[:] if classes is None else classes

    @property
    def df(self):
        return self._df.copy()

    @df.setter
    def df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError(type(value))

        diff = set(self._COLS) - set(value.columns)
        if diff:
            raise ValueError('col {} not in df'.format(sorted(diff)))

        self._df = value

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def frame2time(frames: np.ndarray, fps: float):
        ms = 1000 * frames / fps
        t0 = np.datetime64(0, 's')

        return t0 + ms.astype('timedelta64[ms]')

    def check_columns(self, cols=None, pair=False):
        for col in cols:
            if col not in self._df.columns:
                raise ValueError(f'{col} 데이터가 없습니다.')

        if pair:
            if not any([x.startswith('Pair') for x in self._df.columns]):
                raise ValueError('Pair 계산 결과가 없습니다.')

    def remove_noise(self,
                     seconds: Optional[float] = 1.0,
                     frames: Optional[int] = None,
                     inplace=True):
        """
        연속적으로 나타나는 frame 수가 기준치 미만인 데이터 삭제

        Parameters
        ----------
        seconds : Optional[float], optional
            기준 시간 [sec]
        frames : Optional[int], optional
            기준 frame 수. `seconds`가 입력된 경우 무시함
        inplace : bool, optional

        Returns
        -------
        TrackData
        """
        if not any([seconds, frames]):
            raise ValueError('노이즈 판단 기준치로 seconds, frames 중 하나를 지정해야 함.')

        if seconds:
            frames = self._fps * seconds

        df0 = self.df.sort_values(by=['class', 'object', 'frame']).reset_index(
            drop=True)

        indices0 = _consecutive(df0['frame'].values, index=True)
        index = np.concatenate([x for x in indices0 if x.size >= frames])
        df = df0.loc[index, :].reset_index(drop=True)

        if inplace:
            self._df = df
            td = self
        else:
            td = TrackData(df=df)

        return td

    def get_objects(self, cls: Union[None, int, tuple]) -> np.ndarray:
        if cls is None:
            return np.unique(self.df['object'])

        if isinstance(cls, int):
            mask = self.df['class'] == cls
        else:
            mask = self.df['class'].isin(cls)

        return np.unique(self.df.loc[mask, ['object']])

    def get_class(self, obj: Optional[int]) -> Union[tuple, int]:
        """
        `obj`를 지정하면 (첫 번째로 df에 저장된) obj의 class를 반환.
        **한 object의 class가 여러개 추정되는 오류도 존재**하므로 이용 주의.

        `obj`가 `None`인 경우 존재하는 모든 class 목록 반환

        Parameters
        ----------
        obj : Optional[int]

        Returns
        -------
        int
        """
        if obj is None:
            return tuple(np.unique(self.df['class']))

        row = tuple(self.df['object']).index(obj)

        return int(self.df.loc[row, 'class'])

    def trace(self, obj: Union[int, List[int]]):
        if isinstance(obj, int):
            mask = self.df['object'] == obj
        else:
            mask = self.df['object'].isin(obj)

        return TrackData(df=self._df.loc[mask, :].copy().reset_index(drop=True),
                         fps=self._fps,
                         classes=self._classes)

    def closest_pairs(self, class1: int, class2: int):
        # TODO `scipy.optimize.linear_sum_assignment` 이용
        objects = (tuple(self.get_objects(class1)),
                   tuple(self.get_objects(class2)))
        mi = ['frame', 'object']
        idxs = pd.IndexSlice

        frames = np.sort(np.unique(self._df['frame']))
        df = self.df.set_index(mi)[['cx', 'cy', 'class']]

        full_index = product(frames, sorted(set(objects[0]) | set(objects[1])))
        df: pd.DataFrame = df.reindex(list(full_index)).sort_index()

        dfp = df.copy()  # pair 정보를 저장할 임시 df
        cst12, cst21 = '_closest12', '_closest21'  # 가장 가까운 object를 표시하는 임시 column
        dfp[cst12] = np.nan
        dfp[cst21] = np.nan
        dfp['PairDistance'] = np.nan

        def _closest(objs1, objs2, class2, col):
            for obj in objs1:
                xy1 = df.loc[idxs[:, obj], :].values[:, :-1]
                xy2 = df.loc[idxs[:, objs2], :].values[:, :-1]

                dist = _distance(xy1.reshape([frames.size, 1, 2]),
                                 xy2.reshape([frames.size,
                                              len(objs2), 2]),
                                 p=2,
                                 axis=2)
                misclassified = df.loc[idxs[:, objs2], 'class'] != class2
                dist[misclassified.values.reshape(dist.shape)] = np.nan

                closest = np.argmin(np.nan_to_num(dist, nan=np.inf), axis=1)
                closest[np.all(np.isnan(dist), axis=1)] = -1
                closest_dist = np.nanmin(dist, axis=1)

                dfp.loc[(idxs[:, obj]), col] = [
                    np.nan if x == -1 else objs2[int(x)] for x in closest
                ]
                dfp.loc[(idxs[:, obj]), 'PairDistance'] = closest_dist

        _closest(objects[0], objects[1], class2, cst12)
        _closest(objects[1], objects[0], class1, cst21)

        def _is_match(idx, row, c1, c2):
            if np.isnan(row[c1]):
                return False
            return (idx[1] == dfp.loc[(idx[0], int(row[c1])), c2])

        # TODO logging
        dfp['_m12'] = [_is_match(i, r, cst12, cst21) for i, r in dfp.iterrows()]
        dfp['_m21'] = [_is_match(i, r, cst21, cst12) for i, r in dfp.iterrows()]

        cols = [f'Pair-{self._classes[x]}' for x in (class1, class2)]
        dfp[cols[1]] = np.where(dfp['_m12'], dfp[cst12], np.nan)
        dfp[cols[0]] = np.where(dfp['_m21'], dfp[cst21], np.nan)
        dfp.loc[np.all(np.isnan(dfp[cols]), axis=1), 'PairDistance'] = np.nan

        self._df.set_index(mi)
        self._df = self._df.join(dfp[cols + ['PairDistance']], on=mi)


class EventEstimator:
    _CC = 'CentroidChange'
    _ID = 'IntensityDifference'

    def __init__(self,
                 video: PathLike,
                 track_data: Union[PathLike, TrackData],
                 fps: Optional[float],
                 classes: Optional[tuple] = None) -> None:
        self._video_path = Path(video)
        self._video_path.stat()

        if fps is None:
            self._fps = FFmpeg(stream=False).fps(self._video_path.as_posix())
            logger.info('FPS: {}', self._fps)
        else:
            self._fps = fps

        if isinstance(track_data, TrackData):
            self._td = track_data
        else:
            self._td = TrackData(df=track_data, fps=self._fps, classes=classes)

        self._capture = None
        self.__current_frame = -1

    def __del__(self):
        capture = getattr(self, '_capture', None)
        if capture is not None and capture.isOpened():
            capture.release()

    @property
    def track_data(self):
        return self._td

    def write_data(self, path: PathLike):
        df = self.track_data.df.sort_values(['frame', 'object'])
        df.to_csv(path, index=False)

    @classmethod
    def from_data(cls,
                  path,
                  video: PathLike,
                  fps: Optional[float],
                  classes: Optional[tuple] = None):
        df = pd.read_csv(path)
        return cls(video=video, track_data=df, fps=fps, classes=classes)

    @property
    def video_capture(self) -> cv.VideoCapture:
        if self._capture is None:
            self._capture = cv.VideoCapture(self._video_path.as_posix())

        return self._capture

    def capture(self, frame: int, gray=True) -> np.ndarray:
        """
        대상 영상의 특정 frame 영상 읽어오기

        Parameters
        ----------
        frame : int
            frame (0부터 시작)
        gray : bool
            `True`이면 회색 영상 반환

        Returns
        -------
        np.ndarray
        """
        if self.__current_frame + 1 != frame:
            self.video_capture.set(cv.CAP_PROP_POS_FRAMES, frame)
        self.__current_frame = frame

        retval, image = self.video_capture.read()
        if not retval:
            raise RuntimeError(
                f'대상 영상 불러오기 실패 (frame {frame}): "{self._video_path}"')

        if gray:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        return image

    def calculate_centroid_change(self):
        # world 좌표를 알 수 없기 때문에 걍 pixel 좌표로 계산
        objects = self.track_data.get_objects(cls=None)
        df = self.track_data.df.set_index(['frame', 'object'])
        df[self._CC] = np.nan
        for obj in tqdm(objects,
                        desc='Calculating centroid change',
                        total=objects.size):
            dfo = df.loc[pd.IndexSlice[:, obj], :]
            x, y, d = np.hsplit(dfo[['cx', 'cy', 'diag']].values, [1, 2])
            cc = (np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2) / d[:-1])

            # 두 frame의 차이로 계산되기 때문에 마지막에 nan 입력
            df.loc[pd.IndexSlice[:, obj], self._CC] = np.append(cc, [np.nan])

        self.track_data.df = df.reset_index()

    def _calculate_intensity_diff(self, df: pd.DataFrame, maxval):
        image1, image2 = None, None
        diffs = []
        for frame, row in tqdm(df.iterrows(),
                               total=df.shape[0],
                               desc='Calculating Intensity Difference'):
            bbox = row[['top', 'left', 'height', 'width']].values

            if np.isnan(bbox[0]):
                image1, image2 = None, None
                diffs.append(None)
                continue

            capture = self.capture(frame=frame - 1)
            image2 = capture[int(bbox[0]):int(bbox[0] + bbox[2]),
                             int(bbox[1]):int(bbox[1] + bbox[3])]

            if image1 is None:
                image1 = image2
                diffs.append(None)
                continue

            diff = intensity_difference(image1=image1,
                                        image2=image2,
                                        maxval=maxval)
            diffs.append(diff)
            image1 = image2

        diffs.pop(0)
        diffs.append(None)  # (i, i+1)로 계산한 값을 i번째 행에 대입

        return diffs

    def calculate_intensity_difference(self,
                                       classes: Optional[List[int]] = None):
        df = self.track_data.df
        if classes is None:
            classes = tuple(np.unique(df['class']))

        frames = np.arange(np.min(df['frame']), np.max(df['frame']) + 1)
        maxval = np.iinfo(self.capture(0).dtype).max
        df_list = []
        for cls in classes:
            dfc = df[df['class'] == cls]
            objects = tuple(np.unique(dfc['object']))
            for obj in objects:
                logger.info(
                    'Calculating Intensity Difference of '
                    'class {} object {} ({})', cls, obj,
                    self.track_data._classes[cls])

                dfo: pd.DataFrame = dfc[dfc['object'] == obj]
                dfo = dfo.set_index('frame').reindex(frames)

                diffs = self._calculate_intensity_diff(df=dfo, maxval=maxval)
                dfo[self._ID] = diffs
                df_list.append(dfo.reset_index()[['frame', 'object', self._ID]])

        df_diff = pd.concat(df_list)
        df = pd.merge(left=df,
                      right=df_diff,
                      how='left',
                      on=['frame', 'object'])

        self.track_data.df = df

    def smoothing(self,
                  window: int,
                  cols: Optional[List[str]] = None,
                  method='median'):
        if cols is None:
            cols = [self._CC, self._ID]
            # TODO pair 데이터도 smooth?
        scols = [x + 'Smooth' for x in cols]

        df = self.track_data.df
        frames = np.arange(np.min(df['frame']), np.max(df['frame']) + 1)
        classes = tuple(np.unique(df['class']))
        df_list = []
        for cls in classes:
            dfc = df[df['class'] == cls]
            objects = tuple(np.unique(dfc['object']))
            for obj in objects:
                dfo: pd.DataFrame = dfc[dfc['object'] == obj]
                dfo = dfo.set_index('frame').reindex(frames)
                dfo[scols] = dfo[cols].rolling(window=window,
                                               min_periods=1,
                                               center=True).aggregate(method)
                df_list.append(dfo.reset_index()[['frame', 'object'] + scols])

        df_smooth = pd.concat(df_list)
        df = pd.merge(left=df,
                      right=df_smooth,
                      how='left',
                      on=['frame', 'object'])

        self.track_data.df = df

    def _plot_st_features(self, df: pd.DataFrame, path):
        has_id = np.any(np.isnan(df[self._ID]))

        value_vars = ['cx', 'cy', 'width', 'height', self._CC]
        col_order = ['Center', 'Size', self._CC]
        if has_id:
            value_vars.append(self._ID)
            col_order.append(self._ID)
        value_vars.extend([x for x in df.columns if 'Smooth' in x])

        dfm = pd.melt(df, id_vars=['frame', 'time'], value_vars=value_vars)

        dfm['ax'] = 'Size'
        dfm.loc[dfm['variable'].str.startswith('c'), 'ax'] = 'Center'
        dfm.loc[dfm['variable'].str.startswith(self._CC), 'ax'] = self._CC
        dfm.loc[dfm['variable'].str.startswith(self._ID), 'ax'] = self._ID

        grid = sns.FacetGrid(data=dfm,
                             col='ax',
                             col_wrap=2,
                             col_order=col_order,
                             sharey=False,
                             despine=False,
                             height=4.5,
                             aspect=16 / 9)

        # TODO NA 구간 표시 (_consecutive로 index 달기?)
        grid.map_dataframe(sns.lineplot,
                           x='time',
                           y='value',
                           hue='variable',
                           alpha=0.9)
        grid.set_titles(col_template='{col_name}')
        for ax in grid.axes_dict.values():
            ax.legend()
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

        grid.savefig(path, dpi=300)
        plt.close(grid.figure)

    def plot_st_features(self, path):
        path = Path(path)
        path.stat()
        if not path.is_dir():
            raise NotADirectoryError(path)

        self.track_data.check_columns(cols=(self._CC, self._ID))
        df = self.track_data.df

        classes = np.unique(df['class'])
        for cls in classes:
            objects = np.unique(df.loc[df['class'] == cls, 'object'])
            for obj in objects:
                logger.info('Spatio-temporal feature plot class {} object {}',
                            cls, obj)

                dfo = df.loc[(df['class'] == cls) & (df['object'] == obj), :]
                p = path.joinpath(
                    f'ST-features_class{cls:03d}_object{obj:03d}.png')

                self._plot_st_features(df=dfo, path=p)


class ConstructionEventEstimator(EventEstimator):

    def _excavator_truck_id(self):
        classes_lower = tuple(x.lower() for x in self.track_data.classes)
        excavator = classes_lower.index('excavator')
        truck = classes_lower.index('truck')

        return excavator, truck

    def calculate_excavator_truck_pair(self):
        excavator, truck = self._excavator_truck_id()
        self.track_data.closest_pairs(class1=excavator, class2=truck)

    def _estimate_status(self,
                         centroid_threshold=0.01,
                         intensity_threshold=0.05,
                         distance_threshold=150) -> pd.DataFrame:
        self.track_data.check_columns(cols=(self._CC, self._ID), pair=True)

        df = self.track_data.df.set_index(['frame', 'object'])

        CC, ID = self._CC + 'Smooth', self._ID + 'Smooth'
        CC = CC if CC in df.columns else self._CC
        ID = ID if ID in df.columns else self._ID
        logger.info('Estimate moving status with {}', CC)
        logger.info('Estimate working status with {}', ID)

        df['CC-Exceeded'] = df[CC] >= centroid_threshold
        df['ID-Exceeded'] = df[ID] >= intensity_threshold

        # excavator-truck 거리 기준 따라 근접 판단
        nearby = df['PairDistance'] <= distance_threshold
        df['Pair-Excavator'] = np.where(nearby, df['Pair-Excavator'], np.nan)
        df['Pair-Truck'] = np.where(nearby, df['Pair-Truck'], np.nan)

        # 근접한 excavator, truck 이동/작업 여부 매칭
        df['Pair-Truck-Moving'] = [
            (
                False if np.isnan(r['Pair-Truck']) else  #
                df.loc[(i[0], r['Pair-Truck']), 'CC-Exceeded'])
            for i, r in df.iterrows()
        ]
        df['Pair-Excavator-Working'] = [
            (
                False if np.isnan(r['Pair-Excavator']) else  #
                (df.loc[(i[0], r['Pair-Excavator']), 'ID-Exceeded']) and
                not df.loc[(i[0], r['Pair-Excavator']), 'CC-Exceeded'])
            for i, r in df.iterrows()
        ]

        df['Status'] = np.where(df['CC-Exceeded'], 'Moving', 'Idle')

        # interaction
        excv, trck = self._excavator_truck_id()
        df.loc[((df['class'] == excv) & df['ID-Exceeded']),
               'Status'] = 'WorkAlone'
        df.loc[((df['class'] == excv) & df['ID-Exceeded'] & df['Pair-Truck'] &
                ~df['Pair-Truck-Moving']), 'Status'] = 'WorkInteract'

        df.loc[((df['class'] == trck) & (df['Status'] == 'Idle') &
                df['Pair-Excavator'] & (df['Pair-Excavator-Working'])),
               'Status'] = 'WorkInteract'

        return df.reset_index()

    def _save_status_video(self, df: pd.DataFrame, path: str):
        shape = (int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH)),
                 int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
        writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*'mp4v'), self._fps,
                                shape)

        frames = np.arange(0, np.max(df['frame']))
        dff = df.set_index('frame')['object']
        dffo = df.set_index(['frame', 'object'])
        classes = self.track_data.classes

        for frame in tqdm(frames, desc='Writing Video', total=frames.size):
            img = self.capture(frame=frame - 1, gray=False)
            annotator = Annotator(img, line_width=2, pil=not ascii)

            try:
                objects = np.unique(dff.loc[frame])
            except KeyError:
                writer.write(img)
                continue

            if not objects.size:
                writer.write(img)
                continue

            for obj in objects:
                row = dffo.loc[(frame, obj)]
                box = row[['left', 'top', 'width', 'height']].values
                box[2:] += box[:2]

                cls = int(row['class'])
                label = '{} {} {}'.format(obj, classes[int(cls)], row['Status'])
                annotator.box_label(box=box,
                                    label=label,
                                    color=colors(cls, True))

            writer.write(annotator.result())

        writer.release()

    def estimate_status(self,
                        path: PathLike,
                        centroid_threshold=0.01,
                        intensity_threshold=0.05,
                        distance_threshold=150,
                        save_video=True):
        path = Path(path)
        if path.is_dir():
            path.stat()
            path = path.joinpath('Status.csv')

        df = self._estimate_status(centroid_threshold=centroid_threshold,
                                   intensity_threshold=intensity_threshold,
                                   distance_threshold=distance_threshold)
        df.sort_values(['frame', 'object'])
        df.to_csv(path, index=False)

        if save_video:
            self._save_status_video(df=df,
                                    path=path.with_suffix('.mp4').as_posix())

        return df


if __name__ == '__main__':
    from EventTime.utils import set_plot_style

    # 그래프 스타일 설정
    set_plot_style(context='paper', palette='Dark2')

    # 대상 동영상 파일
    video_path = r'D:\test\EventTime\sample\Daegu22_20210504_023700-023900_fps2.0.mp4'

    # DeepSORT를 통한 추적 결과 파일
    txt_path = r'D:\test\EventTime\sample\Daegu22_20210504_023700-023900_fps2.0.txt'

    # 저장 위치
    out_path = r'D:\test\EventTime\sample\Daegu22_20210504_023700-023900_fps2.0'
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    # 동영상의 FPS
    # (`None`으로 입력 시 컴퓨터에 설치된 `FFmpeg` 프로그램을 통해 추정)
    fps = None

    estimator = ConstructionEventEstimator(video=video_path,
                                           track_data=txt_path,
                                           fps=fps,
                                           classes=None)

    # 노이즈 제거 (1초 이상 연속되지 않는 추적 데이터 제거)
    estimator.track_data.remove_noise(seconds=1)

    # 거리를 기준으로 굴착기-트럭 쌍 생성
    estimator.calculate_excavator_truck_pair()

    # Centroid Change 계산
    estimator.calculate_centroid_change()

    # Intensity Difference 계산 (굴착기 (class 4)만 대상)
    estimator.calculate_intensity_difference(classes=[4])

    # Smoothing (이동 중위수)
    estimator.smoothing(window=10, method='median')

    # 데이터 저장
    estimator.write_data(path=out_path.joinpath('data.csv'))

    # 장비별 분석 그래프 저장
    # (시간별 위치, bounding box 크기, Centroid Change, Intensity Difference)
    estimator.plot_st_features(path=out_path)

    # 공종 인식 및 저장
    # 데이터별로 임계치 조절 필요
    # (centroid_threshold, intensity_threshold, distance_threshold)
    estimator.estimate_status(path=out_path,
                              centroid_threshold=0.02,
                              intensity_threshold=0.04,
                              distance_threshold=150,
                              save_video=True)
