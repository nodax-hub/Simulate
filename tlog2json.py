#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["pymavlink>=2.4.41", "numpy>=1.26"]
# requires-python = ">=3.10"
# ///

"""
Примеры:
  uv run tlog2json.py ./Telemetry/flight.tlog
  uv run tlog2json.py ./Telemetry -r -p "*.tlog" -o ./out
  uv run tlog2json.py ./flight.tlog --json-only
  uv run tlog2json.py ./flight.tlog --ndjson-only --bytes-mode list
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from array import array
from pathlib import Path
from typing import Iterable

import numpy as np
from pymavlink import mavutil


def encode_bytes(obj: bytes | bytearray | memoryview, mode: str):
    b = obj if isinstance(obj, (bytes, bytearray)) else obj.tobytes()
    if mode == "list":
        return list(b)  # массив 0..255
    return {"__bytes__": "base64", "data": base64.b64encode(b).decode("ascii")}


def sanitize(o, bytes_mode: str):
    # Приводим объект к JSON-совместимому виду (рекурсивно)
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, (bytes, bytearray, memoryview)):
        return encode_bytes(o, bytes_mode)
    if isinstance(o, array):
        return list(o)
    if isinstance(o, np.generic):  # np.int64, np.float64 и т.п.
        return o.item()
    if isinstance(o, (list, tuple, set)):
        return [sanitize(x, bytes_mode) for x in o]
    if isinstance(o, dict):
        return {str(k): sanitize(v, bytes_mode) for k, v in o.items()}
    return str(o)  # неизвестный тип — переводим в строку


def tlog_to_ndjson(
        in_path: Path,
        out_ndjson: Path,
        *,
        bytes_mode: str = "base64",
        dialect: str = "common",
        robust_parsing: bool = False,
        none_limit: int = 200,
) -> None:
    m = mavutil.mavlink_connection(
        str(in_path),
        dialect=dialect,
        robust_parsing=robust_parsing,
    )
    
    out_ndjson.parent.mkdir(parents=True, exist_ok=True)
    
    none_in_row = 0
    with out_ndjson.open("w", encoding="utf-8") as f:
        while True:
            msg = m.recv_match(blocking=False)
            if msg is None:
                none_in_row += 1
                if none_in_row >= none_limit:
                    break
                continue
            none_in_row = 0
            
            # базовый dict от pymavlink
            try:
                d = msg.to_dict()
            except AttributeError:
                d = {k: v for k, v in msg.__dict__.items() if not k.startswith("_")}
            
            # добавляем тип и timestamp (если доступен)
            d["_type"] = msg.get_type()
            ts = getattr(msg, "_timestamp", None)
            if ts is None:
                tb = getattr(msg, "time_boot_ms", None)
                ts = (tb / 1000.0) if tb is not None else None
            d["_timestamp"] = ts
            
            f.write(json.dumps(sanitize(d, bytes_mode), ensure_ascii=False) + "\n")


def ndjson_to_json(in_ndjson: Path, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with in_ndjson.open("r", encoding="utf-8") as fin, out_json.open(
            "w", encoding="utf-8"
    ) as fout:
        fout.write("[\n")
        first = True
        for line in fin:
            if not line.strip():
                continue
            if not first:
                fout.write(",\n")
            fout.write(line.rstrip())
            first = False
        fout.write("\n]\n")


def iter_tlog_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def build_output_paths(
        src_file: Path, out_dir_or_file: Path | None, produce_json: bool
) -> tuple[Path, Path | None]:
    """
    Возвращает (путь_ndjson, путь_json_или_None).
    Логика:
      - Если вход один файл и --out ведёт на файл с .ndjson — используем его как NDJSON.
      - Если --out ведёт на директорию (или не задан) — кладём в эту директорию с именем <basename>.ndjson/json.
    """
    if out_dir_or_file is None:
        base = src_file.with_suffix("")  # убираем .tlog
        out_ndjson = src_file.parent / (base.name + ".ndjson")
        out_json = src_file.parent / (base.name + ".json") if produce_json else None
        return out_ndjson, out_json
    
    if out_dir_or_file.suffix.lower() == ".ndjson" and src_file.is_file():
        out_ndjson = out_dir_or_file
        out_json = out_dir_or_file.with_suffix(".json") if produce_json else None
        return out_ndjson, out_json
    
    # иначе считаем, что это директория
    out_dir = out_dir_or_file
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ndjson = out_dir / (src_file.stem + ".ndjson")
    out_json = (out_dir / (src_file.stem + ".json")) if produce_json else None
    return out_ndjson, out_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Преобразование MAVLink .tlog в NDJSON и (опционально) JSON."
    )
    p.add_argument(
        "input",
        type=Path,
        help="Путь к .tlog или каталогу с логами.",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help=(
            "Путь вывода: файл .ndjson (для одиночного входного файла) "
            "или директория (для каталога/нескольких файлов). По умолчанию — рядом с исходником."
        ),
    )
    p.add_argument(
        "-p",
        "--pattern",
        default="*.tlog",
        help="Шаблон для поиска файлов в каталоге (по умолчанию: *.tlog).",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Рекурсивная обработка каталога.",
    )
    p.add_argument(
        "--bytes-mode",
        choices=["base64", "list"],
        default="base64",
        help="Как сериализовать байтовые поля (по умолчанию base64).",
    )
    p.add_argument(
        "--dialect",
        default="common",
        help="MAVLink dialect (по умолчанию common).",
    )
    p.add_argument(
        "--robust-parsing",
        action="store_true",
        help="Включить robust_parsing для pymavlink.",
    )
    p.add_argument(
        "--none-limit",
        type=int,
        default=200,
        help="Сколько подряд None-сообщений ждём до останова (по умолчанию 200).",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--ndjson-only",
        action="store_true",
        help="Только NDJSON (не собирать JSON-массив).",
    )
    g.add_argument(
        "--json-only",
        action="store_true",
        help="Только JSON (сначала будет сгенерирован NDJSON).",
    )
    
    args = p.parse_args(argv)
    
    # Определяем режимы вывода
    if args.ndjson_only and args.json_only:
        p.error("Нельзя одновременно --ndjson-only и --json-only")
    
    produce_ndjson = not args.json_only
    produce_json = not args.ndjson_only
    
    in_path: Path = args.input
    if not in_path.exists():
        p.error(f"Входной путь не найден: {in_path}")
    
    files = list(iter_tlog_files(in_path, args.pattern, args.recursive))
    if not files:
        print("Нет файлов для обработки.", file=sys.stderr)
        return 2
    
    multi_input = len(files) > 1 or in_path.is_dir()
    
    for f in files:
        if not f.is_file():
            continue
        out_ndjson, out_json = build_output_paths(
            f, args.out if multi_input else args.out, produce_json
        )
        
        # NDJSON (как итог или как источник для JSON)
        if produce_ndjson or args.json_only:
            tlog_to_ndjson(
                f,
                out_ndjson,
                bytes_mode=args.bytes_mode,
                dialect=args.dialect,
                robust_parsing=args.robust_parsing,
                none_limit=args.none_limit,
            )
            print(f"OK: {out_ndjson}")
        
        # JSON-массив при необходимости
        if produce_json:
            if out_json is None:
                out_json = out_ndjson.with_suffix(".json")
            ndjson_to_json(out_ndjson, out_json)
            print(f"OK: {out_json}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
