import logging
from pathlib import Path
from tqdm import tqdm
import requests

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler())


def url_exists(url: str) -> bool:
    """
    Check an url exist (or is accessible from where the command is launched.

    :param url: url to check
    :return: boolean True if url exist, False if not.
    """
    try:
        logger.info(f"Checking existance of: {url}")
        res = requests.get(url, timeout=1)
    except requests.exceptions.RequestException as e:
        logger.error(e)
        return False
    return res.status_code == 200


def download_from_url(url: str, dst: Path) -> int:
    """
    Download file from an url with progressbar.

    :param url: url to download file
    :param dst: directory where to put the file

    :return: file_size in bytes.
    """

    if dst.exists():
        raise FileExistsError(f"{dst.absolute()} file already exists.")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Download file from {} in {}.".format(url, dst.as_posix()))

    file_size = int(requests.head(url, allow_redirects=True).headers["Content-Length"])

    logger.info("File size is {} bytes.".format(str(file_size)))

    header = {
        "Range": f"bytes=0-{file_size}",
    }

    logger.info("Begin download and write file.")

    progress_bar = tqdm(
        total=file_size, initial=0, unit="B", unit_scale=True, desc=url.split("/")[-1],
    )

    req = requests.get(url, headers=header, stream=True)

    with dst.open("ab") as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(1024)

    progress_bar.close()

    logger.info("File downloaded.")

    return file_size
