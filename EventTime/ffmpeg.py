import subprocess as sp
import sys

from pathlib import Path
from typing import Union

import click
from loguru import logger

from EventTime.utils import console, set_logger


def _str_args(args: Union[str, list]):
    if isinstance(args, str):
        return args

    return ' '.join((str(x) for x in args))


class nullcontext:

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


class FFmpeg:

    def __init__(self, path=None, stream=True, status=True) -> None:
        self._path = ('ffmpeg', 'ffprobe')

        if path is not None:
            path = Path(path).resolve()
            path.stat()
            self._path = tuple(path.joinpath(x).as_posix() for x in self._path)

        if stream:
            self._stream = {}
        else:
            self._stream = dict(stdout=sp.DEVNULL, stderr=sp.PIPE)
        self._status = status

    def ffmpeg(self, args: Union[str, list], stream=None):
        if stream is None:
            stream = self._stream

        sargs = _str_args(args)
        logger.debug('ffmpeg {}', sargs)

        cm = (console.status('Running ffmpeg...')
              if stream and self._status else nullcontext())
        with cm:
            out = sp.run(f'{self._path[0]} {sargs}', **stream)

        return out

    def ffprobe(self, args: Union[str, list]):
        sargs = _str_args(args)
        logger.debug('ffprobe {}', sargs)

        return sp.run(f'{self._path[1]} {sargs}',
                      stdout=sp.PIPE,
                      stderr=sp.STDOUT)

    def duration(self, src):
        args = ('-v error -show_entries format=duration -of '
                f'default=noprint_wrappers=1:nokey=1 "{src}"')
        out = self.ffprobe(args=args)

        try:
            d = float(out.stdout)
        except ValueError:
            raise RuntimeError(out.stdout)

        return d

    def fps(self, src) -> float:
        args = ('-v error -select_streams v -of '
                'default=noprint_wrappers=1:nokey=1 '
                f'-show_entries stream=r_frame_rate "{src}"')
        out = self.ffprobe(args=args)
        out_str = out.stdout.decode('utf-8')
        fps_frac = out_str.strip().split('/')

        try:
            assert len(fps_frac) == 2
            fps = float(fps_frac[0]) / float(fps_frac[1])
        except (AssertionError, ValueError):
            raise ValueError(out_str)

        return fps

    def cut(self, src, dst, ss, to, fps=None):
        fps_ = '' if fps is None else f'-r {fps} '
        args = f'-ss {ss} -to {to} -i "{src}" {fps_}"{dst}"'
        return self.ffmpeg(args=args)

    def change_fps(self, src, dst, fps: float):
        args = f'-i "{src}" -filter:v fps={fps} "{dst}"'
        return self.ffmpeg(args=args)

    def capture(self, src, dst, ss):
        args = f'-ss {ss} -i "{src}" -frames:v 1 -q:v 2 "{dst}"'
        return self.ffmpeg(args=args)


def _dst_path(src: Path, dst_dir: Path, ext=None, ss=None, to=None, fps=None):
    ss_ = str(ss).replace(':', '')
    to_ = str(to).replace(':', '')
    fps_ = '' if fps is None else f'_fps{fps}'
    if ext is None:
        ext = src.suffix

    return dst_dir.joinpath(f'{src.stem}_{ss_}-{to_}{fps_}{ext}')


def prep_path(src, dst, ext=None, ss=None, to=None, fps=None):
    src = Path(src)
    dst = src.parent if dst is None else Path(dst)
    if dst.is_dir():
        dst = _dst_path(src=src, dst_dir=dst, ext=ext, ss=ss, to=to, fps=fps)

    if dst.exists():
        logger.warning('dst already exists: "{}"', dst)
    else:
        logger.info('dst: "{}"', dst)

    return src, dst


@click.group()
@click.option('--ffmpeg')
@click.option('--stream/--no-stream', default=True)
@click.option('--loglevel', default='INFO')
@click.pass_context
def cli(ctx, ffmpeg, stream, loglevel):
    set_logger(level=loglevel)
    logger.debug('sys.argv: "{}"', ' '.join(sys.argv))

    ctx.obj = FFmpeg(path=ffmpeg, stream=stream)


argument_src = click.argument('src',
                              type=click.Path(exists=True, dir_okay=False))
argument_dst = click.argument('dst', type=click.Path(), required=False)


@cli.command()
@click.option('--ss')
@click.option('--to')
@click.option('--fps', type=float)
@argument_src
@argument_dst
@click.pass_obj
def cut(ffmpeg: FFmpeg, ss, to, fps, src, dst):
    src, dst = prep_path(src=src, dst=dst, ss=ss, to=to, fps=fps)
    ffmpeg.cut(src=src, dst=dst, ss=ss, to=to, fps=fps)


@cli.command()
@click.option('--fps', type=float, required=True)
@argument_src
@argument_dst
@click.pass_obj
def fps(ffmpeg: FFmpeg, src, dst, fps):
    src, dst = prep_path(src=src, dst=dst, fps=fps)
    ffmpeg.change_fps(src=src, dst=dst, fps=fps)


@cli.command()
@click.option('--ss', required=True)
@argument_src
@argument_dst
@click.pass_obj
def capture(ffmpeg: FFmpeg, src, dst, ss):
    src, dst = prep_path(src=src, dst=dst, ext='.jpg', ss=ss, fps=fps)
    ffmpeg.capture(src=src, dst=dst, ss=ss)


if __name__ == '__main__':
    cli()
