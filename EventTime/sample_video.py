import random
from datetime import timedelta
from pathlib import Path

import click
from loguru import logger

from EventTime.ffmpeg import FFmpeg, fps, prep_path
from EventTime.utils import set_logger, track


class FFmpegSampler(FFmpeg):

    @staticmethod
    def _td_str(td: timedelta):
        s = str(td)
        if len(s) == 8:  # HH:MM:SS
            return s
        elif len(s) == 7:  # H:MM:SS
            return '0' + s
        else:
            raise ValueError(f'Unexpected formate: {s}')

    def sample_time(self, src, sample_range=(0.1, 0.9), steps=1, delta=5):
        duration = self.duration(src)
        start = timedelta(seconds=random.randint(
            int(duration * sample_range[0]),
            int(duration * sample_range[1]),
        ))
        samples = [start + timedelta(seconds=(x * delta)) for x in range(steps)]
        logger.debug('samples: {}', samples)

        return samples

    def sample_image(self, src: Path, dst: Path, samples):
        for sample in samples:
            ss = self._td_str(sample)
            assert len(ss) == 8  # HH:MM:SS
            ts = ss.replace(':', '-')

            path = dst.joinpath(f'{src.stem}_{ts}.jpg')
            self.capture(src=src.as_posix(), dst=path.as_posix(), ss=sample)
            logger.info('Saved sample "{}"', path)

    def sample_video(self, src, dst, sample, duration, fps):
        ss = sample
        to = sample + timedelta(minutes=duration)
        src_, dst_ = prep_path(src=src, dst=dst, ss=ss, to=to, fps=fps)
        self.cut(src=src_, dst=dst_, ss=ss, to=to, fps=fps)


@click.group()
def cli():
    set_logger()


@cli.command()
@click.option('--seed', type=int)
@click.option('--range', 'sample_range', nargs=2, default=(0.1, 0.9))
@click.option('--steps', default=1)
@click.option('--delta', default=5)
@click.option('--ext', default='.mp4')
@click.option('--ffmpeg-dir')
@click.argument('src', type=click.Path(exists=True))
@click.argument('dst', type=click.Path(exists=True, file_okay=False))
def image(seed, sample_range, steps, delta, ext, ffmpeg_dir, src, dst):
    logger.info('Seed: {}', seed)
    if seed:
        random.seed(seed)

    src = Path(src)
    dst = Path(dst)

    if src.is_file():
        total = 1
        files = [src]
    else:
        total = sum(1 for _ in src.glob(f'*{ext}'))
        files = src.glob(f'*{ext}')

    fs = FFmpegSampler(path=ffmpeg_dir, stream=False)

    for file in track(files, description='Sampling...', total=total):
        samples = fs.sample_time(src=file.as_posix(),
                                 sample_range=sample_range,
                                 steps=steps,
                                 delta=delta)
        fs.sample_image(src=file, dst=dst, samples=samples)


@cli.command()
@click.option('--seed', type=int)
@click.option('--range', 'sample_range', nargs=2, default=(0.1, 0.9))
@click.option('--duration', default=5)
@click.option('--fps', default=2)
@click.option('--ext', default='.mp4')
@click.option('--ffmpeg-dir')
@click.argument('src', type=click.Path(exists=True))
@click.argument('dst', type=click.Path(exists=True, file_okay=False))
def video(seed, sample_range, duration, fps, ext, ffmpeg_dir, src, dst):
    logger.info('Seed: {}', seed)
    if seed:
        random.seed(seed)

    src = Path(src)
    dst = Path(dst)

    if src.is_file():
        total = 1
        files = [src]
    else:
        total = sum(1 for _ in src.glob(f'*{ext}'))
        files = src.glob(f'*{ext}')

    fs = FFmpegSampler(path=ffmpeg_dir, stream=False, status=False)
    for file in track(files, description='Sampling...', total=total):
        sample = fs.sample_time(src=file.as_posix(),
                                sample_range=sample_range,
                                steps=1)[0]
        fs.sample_video(src=file,
                        dst=dst,
                        sample=sample,
                        duration=duration,
                        fps=fps)


if __name__ == '__main__':
    cli()
