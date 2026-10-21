# -*- coding: utf-8 -*-
"""
プロジェクトのフォルダ構成を、ファイル数と容量つきで一覧にする(整理の準備用)。
画像が何千枚もあるフォルダは、中身を1つずつ出さずに「bmp 2900枚」のようにまとめる。

使い方(プロジェクトのいちばん上のフォルダで):
    python list_project.py > project_tree.txt
"""
import os
import sys
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
SKIP = {".git", "__pycache__", ".venv", "venv", ".idea", ".vscode"}
MANY = 20   # これより多いファイルは、拡張子ごとの数だけ表示する


def size_str(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def walk(path, depth):
    try:
        entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        return 0
    files = [e for e in entries if e.is_file()]
    dirs = [e for e in entries if e.is_dir() and e.name not in SKIP]
    total = sum(e.stat().st_size for e in files)
    lines = []
    for d in dirs:
        sub_total, sub_lines = walk(d.path, depth + 1)
        total += sub_total
        lines.append(("  " * (depth + 1)) + f"[{d.name}/]  ({size_str(sub_total)})")
        lines.extend(sub_lines)
    if len(files) > MANY:
        cnt = Counter(os.path.splitext(f.name)[1].lower() or "(拡張子なし)" for f in files)
        summary = ", ".join(f"{ext} {n}個" for ext, n in cnt.most_common())
        lines.append(("  " * (depth + 1)) + f"... {summary}  (合計 {size_str(sum(f.stat().st_size for f in files))})")
    else:
        for f in files:
            st = f.stat()
            import time
            t = time.strftime("%Y-%m-%d", time.localtime(st.st_mtime))
            lines.append(("  " * (depth + 1)) + f"{f.name}  ({size_str(st.st_size)}, {t})")
    return total, lines


if __name__ == "__main__":
    total, lines = walk(ROOT, 0)
    print(f"{os.path.abspath(ROOT)}  (合計 {size_str(total)})")
    print("\n".join(lines))