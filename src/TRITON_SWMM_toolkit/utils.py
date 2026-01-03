import json
from pathlib import Path
from collections import defaultdict
from typing import Union, Iterable, Optional
import os


def print_json_file_tree(
    json_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    *,
    show_missing: bool = True,
) -> None:
    """
    Print a directory tree of file paths found in a JSON file.

    If base_dir is None, the common root directory is auto-detected.

    Parameters
    ----------
    json_path : str | Path
        Path to the JSON file containing file paths.
    base_dir : str | Path | None
        Common base directory shared by all files.
        If None, the root is auto-detected.
    show_missing : bool, optional
        If False, paths that do not exist on disk are skipped.
    """

    json_path = Path(json_path)

    with json_path.open("r") as f:
        data = json.load(f)

    # -------------------------
    # Extract paths recursively
    # -------------------------
    def is_path_like(s: str) -> bool:
        return os.path.isabs(s) or "\\" in s or "/" in s or Path(s).suffix != ""

    def extract_paths(obj) -> list[Path]:
        if isinstance(obj, str) and is_path_like(obj):
            return [Path(obj).expanduser()]
        elif isinstance(obj, dict):
            paths = []
            for v in obj.values():
                paths.extend(extract_paths(v))
            return paths
        elif isinstance(obj, list):
            paths = []
            for v in obj:
                paths.extend(extract_paths(v))
            return paths
        return []

    paths = extract_paths(data)
    if not paths:
        print("(no paths found)")
        return

    paths = [p.resolve() for p in paths]

    # -------------------------
    # Auto-detect base directory
    # -------------------------
    if base_dir is None:
        common_parts = list(zip(*(p.parts for p in paths)))
        root_parts = []
        for parts in common_parts:
            if len(set(parts)) == 1:
                root_parts.append(parts[0])
            else:
                break
        base_dir = Path(*root_parts)

    base_dir = Path(base_dir)

    # -------------------------
    # Build directory tree
    # -------------------------
    def build_tree(paths: Iterable[Path]):
        tree = lambda: defaultdict(tree)
        root = tree()

        for p in paths:
            try:
                rel = p.relative_to(base_dir)
            except ValueError:
                continue

            current = root
            for part in rel.parts:
                current = current[part]

        return root

    def print_tree(tree, prefix=""):
        items = list(tree.items())
        for i, (name, subtree) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            print(prefix + connector + name)
            extension = "    " if is_last else "│   "
            print_tree(subtree, prefix + extension)

    tree = build_tree(paths)

    print(base_dir)
    print_tree(tree)
