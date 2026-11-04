from pathlib import Path
import requests, json, hashlib, zipfile, io, struct, zlib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
api = 'https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:10.5683/SP3/RPCWBY'
r = requests.get(api, timeout=60)
r.raise_for_status()
meta = r.json()
(RAW / 'dataset_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
files = meta['data']['latestVersion']['files']
record = next(x['dataFile'] for x in files if x['dataFile']['filename'] == '1_Test_1.zip')
member = '1_Test#1/JC_Charac_chanel1_CC_Cycle389_March22_Channel_1_Wb_1.CSV'
url = f"https://borealisdata.ca/api/access/datafile/{record['id']}"
def getrange(start, end):
    r = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=60)
    r.raise_for_status()
    assert r.status_code == 206 and len(r.content) == end-start+1
    return r.content
tailstart = record['filesize'] - 131072
tail = getrange(tailstart, record['filesize']-1)
(RAW / 'zip_central_directory.bin').write_bytes(tail)
with zipfile.ZipFile(io.BytesIO(tail)) as z:
    info = z.getinfo(member)
    start = info.header_offset + tailstart
    head = getrange(start, start+29)
    nname, nextra = struct.unpack_from('<HH', head, 26)
    start += 30+nname+nextra
    payload = bytearray()
    for pos in range(start, start+info.compress_size, 1048576):
        payload.extend(getrange(pos, min(pos+1048576, start+info.compress_size)-1))
        print(f'Downloaded {len(payload)}/{info.compress_size} compressed bytes', flush=True)
    raw = zlib.decompress(payload, -15)
    assert len(raw) == info.file_size and zlib.crc32(raw) == info.CRC
    (RAW / Path(member).name).write_bytes(raw)
    (RAW / 'download_provenance.json').write_text(json.dumps({
        'dataset_doi':'10.5683/SP3/RPCWBY', 'url':url, 'member':member,
        'download_method':'HTTP byte ranges from original ZIP, no data modification',
        'size':len(raw), 'crc32':f'{info.CRC:08x}', 'crc32_verified':True,
        'sha256':hashlib.sha256(raw).hexdigest(), 'archive_checksum_verified':False,
        'archive_published_checksum':record['checksum']}, indent=2), encoding='utf-8')
readme = requests.get('https://borealisdata.ca/api/access/datafile/1143918', timeout=60)
readme.raise_for_status()
(RAW/'0_readme.txt').write_bytes(readme.content)
print('Raw CSV extracted', flush=True)
