import sys
from typing import Optional, Union

import matplotlib as mpl
import seaborn as sns
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track as _track

console = Console()


def set_logger(level: Union[int, str, None] = None):
    logger.remove()

    if level is None:
        if any('debug' in x.lower() for x in sys.argv):
            level = 'DEBUG'
        else:
            level = 'INFO'

    rich_handler = RichHandler(console=console, log_time_format='[%X]')
    logger.add(rich_handler, level=level, format='{message}', enqueue=True)
    logger.add('EventTime.log',
               level='DEBUG',
               rotation='1 week',
               retention='1 month',
               encoding='UTF-8-SIG',
               enqueue=True)


def track(sequence,
          description='Working...',
          total: Optional[float] = None,
          transient=False,
          **kwargs):
    """Track progress on console by iterating over a sequence."""
    return _track(sequence=sequence,
                  description=description,
                  total=total,
                  console=console,
                  transient=transient,
                  **kwargs)


def set_plot_style(context='notebook',
                   palette='deep',
                   font='sans-serif',
                   font_scale=1.0):
    mpl.rcParams['axes.unicode_minus'] = False
    sns.set_theme(context=context,
                  style='whitegrid',
                  palette=palette,
                  font=font,
                  font_scale=font_scale,
                  rc={
                      'axes.edgecolor': '0.2',
                      'grid.color': '0.8'
                  })
