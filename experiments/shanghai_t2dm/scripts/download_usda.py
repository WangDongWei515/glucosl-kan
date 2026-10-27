from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import shutil
import stat


STAGE = Path(__file__).resolve().parents[3]
ROOT = STAGE / 'data/raw/usda_fdc/sr_legacy_2018-04'
ARCHIVE_URL = 'https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip'
ARCHIVE_SHA256 = 'b80817294b8850530aaedf2e515c02593b1824f763a0ff356e5c2081643e6fd0'


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = ROOT / 'FoodData_Central_sr_legacy_food_csv_2018-04.zip'
    if not archive_path.exists():
        partial = archive_path.with_name(archive_path.name + '.download')
        print('Downloading SR Legacy CSV', flush=True)
        with urlopen(ARCHIVE_URL, timeout=120) as response, partial.open('wb') as destination:
            shutil.copyfileobj(response, destination, length=1024 * 1024)
        partial.replace(archive_path)
    digest = digest_file(archive_path)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f'Existing raw file has an unexpected checksum: {archive_path}')
    extracted = ROOT / 'extracted'
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        for entry in archive.infolist():
            member = PurePosixPath(entry.filename)
            if member.is_absolute() or '..' in member.parts or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(entry.filename)
        if not extracted.exists():
            extracted.mkdir()
            archive.extractall(extracted)
    print('SR Legacy archive SHA-256', digest, '- FoodData Central data are public domain (CC0 1.0); cite FoodData Central as the source.')


if __name__ == '__main__':
    main()
