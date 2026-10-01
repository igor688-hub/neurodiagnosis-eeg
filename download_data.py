import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = os.environ.get("EEG_DATA_URL", "").rstrip("/") + "/"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_XML_PATH = DEFAULT_DATA_DIR / "data.xml"


S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _parse_contents(root):
    return [
        (c.find("s3:Key", S3_NS).text, int(c.find("s3:Size", S3_NS).text))
        for c in root.findall(".//s3:Contents", S3_NS)
    ]


def fetch_remote_listing():
    """Full listing of the storage bucket."""
    items, token = [], None
    while True:
        url = BASE_URL + "?list-type=2"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            root = ET.fromstring(resp.read())
        items.extend(_parse_contents(root))
        truncated = root.find("s3:IsTruncated", S3_NS)
        if truncated is None or truncated.text != "true":
            return items
        token = root.find("s3:NextContinuationToken", S3_NS).text


def load_file_list(xml_path: Path, remote: bool = False):
    """File keys and sizes."""
    if remote or not xml_path.exists():
        print("Fetching the current bucket listing...")
        return fetch_remote_listing()
    return _parse_contents(ET.parse(xml_path).getroot())


def filter_files(items, group_filter=None, limit_subjects=None):
    """Files filtered by group and number of subjects per group."""
    by_group_subject = {}
    for key, size in items:
        parts = key.split("/")
        if len(parts) != 3:
            continue
        group, subject, fname = parts
        if group_filter and group != group_filter:
            continue
        key_tuple = (group, subject)
        if key_tuple not in by_group_subject:
            by_group_subject[key_tuple] = []
        by_group_subject[key_tuple].append((key, size))

    selected_files = []
    group_counts = {}
    for (group, subject), files in by_group_subject.items():
        count = group_counts.get(group, 0)
        if limit_subjects is not None and count >= limit_subjects:
            continue
        group_counts[group] = count + 1
        selected_files.extend(files)

    return selected_files


def download_single_file(item, output_dir: Path):
    """Download one file and check its size."""
    key, expected_size = item
    target_path = output_dir / key

    if target_path.exists() and target_path.stat().st_size == expected_size:
        return "skipped", key, expected_size

    target_path.parent.mkdir(parents=True, exist_ok=True)

    encoded_key = urllib.parse.quote(key)
    url = BASE_URL + encoded_key

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) != expected_size:
                raise IOError(f"got {len(data)} bytes, expected {expected_size}")

            temp_path = target_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                f.write(data)
            temp_path.replace(target_path)
            return "downloaded", key, len(data)
        except Exception as e:
            if attempt == 2:
                return "failed", key, str(e)
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="EEG dataset downloader")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_DATA_DIR, help="where to save the data")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH, help="path to data.xml")
    parser.add_argument("--group", "-g", type=str, default=None, help="group folder to download")
    parser.add_argument("--limit", "-l", type=int, default=None, help="maximum number of subjects per group")
    parser.add_argument("--threads", "-t", type=int, default=12, help="parallel downloads")
    parser.add_argument("--remote", action="store_true", help="use the current bucket listing instead of data.xml")
    args = parser.parse_args()

    if BASE_URL == "/":
        sys.exit("Set EEG_DATA_URL to the dataset address provided by the organisers.")

    items = load_file_list(args.xml_path, remote=args.remote)
    files_to_download = filter_files(items, group_filter=args.group, limit_subjects=args.limit)
    total_size = sum(sz for _, sz in files_to_download)
    print(f"Files to download: {len(files_to_download)} of {len(items)} ({total_size / (1024 * 1024):.2f} MB) -> {args.output_dir.resolve()}")

    downloaded = 0
    skipped = 0
    failed = 0
    total = len(files_to_download)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(download_single_file, item, args.output_dir): item for item in files_to_download}
        for future in as_completed(futures):
            status, key, info = future.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"[ERROR] {key}: {info}")

            done = downloaded + skipped + failed
            pct = done / total * 100
            sys.stdout.write(f"\rProgress: {done}/{total} ({pct:.1f}%) | downloaded: {downloaded} | skipped: {skipped} | failed: {failed}")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f} s: downloaded {downloaded}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
