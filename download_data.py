"""
Скрипт для скачивания датасета соревнования по диагностике ПТСР по ЭЭГ.
Данные скачиваются из бакета Yandex Cloud: https://example.invalid/

Использование:
    python download_data.py                  # Скачать все 996 файлов (~77 МБ) в data/
    python download_data.py --limit 3        # Скачать только по 3 испытуемых из каждой группы (быстрый старт)
    python download_data.py --group ПТСР     # Скачать только определенную группу
"""

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://example.invalid/"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_XML_PATH = DEFAULT_DATA_DIR / "data.xml"


def load_file_list(xml_path: Path):
    """Извлекает список файлов и их размеры из data.xml"""
    if not xml_path.exists():
        print(f"Файл {xml_path} не найден. Загружаем листинг бакета напрямую...")
        req = urllib.request.Request(BASE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
        root = ET.fromstring(content)
    else:
        tree = ET.parse(xml_path)
        root = tree.getroot()

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    items = []
    for c in root.findall(".//s3:Contents", ns):
        key = c.find("s3:Key", ns).text
        size = int(c.find("s3:Size", ns).text)
        items.append((key, size))
    return items


def filter_files(items, group_filter=None, limit_subjects=None):
    """Фильтрация файлов по группе и лимиту испытуемых."""
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
    """Скачивает один файл с проверкой размера."""
    key, expected_size = item
    target_path = output_dir / key

    # Проверка: если файл уже скачан целиком, пропускаем
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
    parser = argparse.ArgumentParser(description="Загрузчик датасета ЭЭГ ПТСР")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_DATA_DIR, help="Куда сохранять данные")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH, help="Путь к data.xml")
    parser.add_argument("--group", "-g", type=str, default=None, help="Фильтр по группе (Норма, ПТСР, Соматоформные)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Ограничение количества испытуемых на группу")
    parser.add_argument("--threads", "-t", type=int, default=12, help="Количество параллельных потоков скачивания")
    args = parser.parse_args()

    print("=" * 60)
    print("Загрузка датасета ЭЭГ: Нейродиагностика ПТСР")
    print("=" * 60)

    items = load_file_list(args.xml_path)
    print(f"Всего файлов в листинге: {len(items)}")

    files_to_download = filter_files(items, group_filter=args.group, limit_subjects=args.limit)
    total_size = sum(sz for _, sz in files_to_download)
    print(f"Файлов к скачиванию: {len(files_to_download)} ({total_size / (1024 * 1024):.2f} МБ)")
    print(f"Папка назначения: {args.output_dir.resolve()}")
    print(f"Потоков: {args.threads}")
    print("-" * 60)

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
                print(f"[ОШИБКА] {key}: {info}")

            done = downloaded + skipped + failed
            pct = done / total * 100
            sys.stdout.write(f"\rПрогресс: {done}/{total} ({pct:.1f}%) | Скачано: {downloaded} | Пропущено: {skipped} | Ошибок: {failed}")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n\nГотово за {elapsed:.1f} с!")
    print(f"Итог: скачано={downloaded}, пропущено (уже были)={skipped}, ошибок={failed}")


if __name__ == "__main__":
    main()
