import os
import platform
from urllib.request import urlretrieve

from tqdm import tqdm


def get_cache_directory() -> str:
    package_dir = 'biospectools_package'
    if platform.system() == 'Windows':
        cache_path = os.environ['LOCALAPPDATA']
    else:
        cache_path = os.path.join(os.environ['HOME'], '.cache')
    return os.path.join(cache_path, package_dir)
os.makedirs(get_cache_directory(), exist_ok=True)  # noqa: E305


def download_http(url, filename, reporthook='tqdm', overwrite='raise'):
    """
    Wraps `urllib.request.urlretrieve`. Adds check for file
    existance and default progressbar for reporthook.

    Parameters
    ----------
    url: `str`
    filename: `str`
        Path to save data
    reporthook: `Callable` or `str`
        'tqdm' for default progressbar, otherwise see urlretrieve docs
    overwrite: `bool` or `str`, default 'raise'
        If ``overwrite`` is 'raise', then raise error in case file exists
        If ``overwrite`` is boolean, then correspondingly will overwrite file

    Returns
    -------
    saved_path: `str`
        Path to the newly created data file
    http_answer: `HTTPMessage`
        Resulting HTTPMessage object
    """
    if filename is None:
        raise ValueError('filename must be not None')

    if os.path.exists(filename) and overwrite == 'raise':
        raise FileExistsError(
            f'Use overwrite=True to overwrite file {filename}')
    elif os.path.exists(filename) and not overwrite:
        return filename, None

    if reporthook == 'tqdm':
        desc = (f'Downloading '
                f'{os.path.split(url)[-1]} '
                f'to: {filename if filename else "temporary file"}')
        with _TqdmAdapted(
                unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                desc=desc) as pbar:
            res = urlretrieve(url, filename, reporthook=pbar.update_to)
            pbar.total = pbar.n
        return res
    else:
        return urlretrieve(url, filename, reporthook)


class _TqdmAdapted(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize
