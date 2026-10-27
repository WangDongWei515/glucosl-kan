from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import shutil
import stat


STAGE = Path(__file__).resolve().parents[3]
ROOT = STAGE / 'data/raw/cgmacros/1.0.0'
FILES = {
    'CGMacros_dateshifted365.zip': '05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d',
    'DataDictionary.pdf': 'f161ff8df4bd95efa79d3daa8aba5eab754df287ab0f4dfc74bce156f6862c1a',
    'DataDictionary_Bio.csv': '2d5b4c27873284ddbf218df51172b69ffa9caf6e97a3a122977b7a2761da4c61',
    'DataDictionary_CGMacros-00X.csv': 'db7bf114d61b90fc5de43fed5e5cd6ba33af37e900a706e1b5747d7b0ffac3f8',
    'DataDictionary_Gut_Health_Test.csv': 'fbc7555aacb93b2fb107b65db8d4a3d88d3546e7818fe5378fd2840cdceeb1b0',
    'DataDictionary_Microbes.csv': 'eb75be859489f59f6b81c3f8d29c307f078498655b4ae0878e1aa49a57d61f76',
    'LICENSE.txt': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for filename, expected in FILES.items():
        path = ROOT / filename
        if not path.exists():
            base = 'https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/' if filename.endswith('.zip') else 'https://physionet.org/files/cgmacros/1.0.0/'
            partial = path.with_name(path.name + '.download')
            print('Downloading', filename, flush=True)
            with urlopen(base + filename, timeout=120) as response, partial.open('wb') as destination:
                shutil.copyfileobj(response, destination, length=1024 * 1024)
            if digest_file(partial) != expected:
                raise ValueError(f'Checksum mismatch for {filename}; partial retained for inspection')
            partial.replace(path)
        if digest_file(path) != expected:
            raise ValueError(f'Existing raw file has an unexpected checksum: {path}')
    archive_path = ROOT / 'CGMacros_dateshifted365.zip'
    extracted = ROOT / 'extracted'
    with ZipFile(archive_path) as archive:
        if not extracted.exists():
            for entry in archive.infolist():
                member = PurePosixPath(entry.filename)
                if member.is_absolute() or '..' in member.parts or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(entry.filename)
            extracted.mkdir()
            archive.extractall(extracted)
        for filename in archive.namelist():
            if filename.lower().endswith('.csv'):
                if hashlib.sha256(archive.read(filename)).hexdigest() != digest_file(extracted / filename):
                    raise ValueError(f'Extracted source differs from archive: {filename}')
    print('Pinned source files verified. License: CC BY-NC-SA 4.0, as stated on the PhysioNet page. The official LICENSE.txt is empty.')


if __name__ == '__main__':
    main()
