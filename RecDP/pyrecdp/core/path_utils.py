from pathlib import Path
from typing import Optional, List, Union


def get_files(
        input_dir: Union[str, Path],
        glob: str = "**/[!.]*",
        exclude: Optional[List] = None,
        exclude_hidden: bool = True,
        recursive: bool = False,
        required_exts: Optional[List[str]] = None
) -> set[Path]:
    """ get files from a directory"""
    if input_dir is None:
        raise ValueError("input_dir is required!")

    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    all_files = set()
    rejected_files = set()

    if exclude is not None:
        for excluded_pattern in exclude:
            if recursive:
                # Recursive glob
                for file in input_dir.rglob(excluded_pattern):
                    rejected_files.add(Path(file))
            else:
                # Non-recursive glob
                for file in input_dir.glob(excluded_pattern):
                    rejected_files.add(Path(file))

    file_refs = list(input_dir.rglob(glob) if recursive else input_dir.glob(glob))

    for ref in file_refs:
        # Manually check if file is hidden or directory instead of
        # in glob for backwards compatibility.
        is_dir = ref.is_dir()
        skip_because_hidden = exclude_hidden and ref.name.startswith(".")
        skip_because_excluded = ref in rejected_files
        skip_because_bad_ext = (
                required_exts is not None and ref.suffix not in required_exts
        )
        if (
                is_dir
                or skip_because_hidden
                or skip_because_bad_ext
                or skip_because_excluded
        ):
            continue
        else:
            ref.absolute().as_posix()
            all_files.add(ref)

    return all_files
