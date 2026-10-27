from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import json
import shutil
import stat


STAGE = Path(__file__).resolve().parents[3]
ROOT = STAGE / 'data/raw/shanghai_t2dm/v3'
ARTICLE_API = 'https://api.figshare.com/v2/articles/20444397/versions/3'
ARCHIVE_URL = 'https://ndownloader.figshare.com/files/38259264'
ARCHIVE_MD5 = 'ecadc819dd20aeb92d60ce41ceb16fd2'
ARCHIVE_SHA256 = '5ea3d97d189edc38de69a73ab5a7b653085c5cda4691b2eabe2a474295e0d05e'


def digest_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def member_path(entry):
    # The archive's top folder name is GBK-encoded Chinese; it is dropped so local paths stay stable.
    member = PurePosixPath(entry.filename)
    if member.is_absolute() or '..' in member.parts or stat.S_ISLNK(entry.external_attr >> 16):
        raise ValueError(entry.filename)
    return PurePosixPath(*member.parts[1:]) if len(member.parts) > 1 else None


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata_path = ROOT / 'source_metadata.json'
    if not metadata_path.exists():
        with urlopen(ARTICLE_API, timeout=60) as response:
            metadata = json.load(response)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    metadata = json.loads(metadata_path.read_text())
    assert metadata['license']['name'] == 'CC BY 4.0'
    assert [item['computed_md5'] for item in metadata['files']] == [ARCHIVE_MD5]
    archive_path = ROOT / 'data.zip'
    if not archive_path.exists():
        partial = archive_path.with_name(archive_path.name + '.download')
        print('Downloading data.zip', flush=True)
        with urlopen(ARCHIVE_URL, timeout=120) as response, partial.open('wb') as destination:
            shutil.copyfileobj(response, destination, length=1024 * 1024)
        if digest_file(partial, 'md5') != ARCHIVE_MD5:
            raise ValueError('Checksum mismatch for data.zip; partial retained for inspection')
        partial.replace(archive_path)
    if digest_file(archive_path, 'md5') != ARCHIVE_MD5 or digest_file(archive_path, 'sha256') != ARCHIVE_SHA256:
        raise ValueError(f'Existing raw file has an unexpected checksum: {archive_path}')
    extracted = ROOT / 'extracted'
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        members = [(entry, member_path(entry)) for entry in archive.infolist()]
        if not extracted.exists():
            extracted.mkdir()
            for entry, target in members:
                if target is None or entry.is_dir():
                    continue
                (extracted / target).parent.mkdir(parents=True, exist_ok=True)
                (extracted / target).write_bytes(archive.read(entry))
        for entry, target in members:
            if target is not None and not entry.is_dir():
                if hashlib.sha256(archive.read(entry)).hexdigest() != digest_file(extracted / target, 'sha256'):
                    raise ValueError(f'Extracted source differs from archive: {target}')
    print('Pinned figshare v3 archive verified. License: CC BY 4.0. Cite Zhao et al. Sci Data 2023;10:35.')


if __name__ == '__main__':
    main()
