from pathlib import Path
import sys
from typing import List

_HELP_TEXT = """
Welcome to the file selection menu.

You may key in numbers to select the file of your choice. Alternatively, some
options made available to you are:
1. "?", "m"
    Shows this help menu
2. ".."
    Goes up one level from current directory.
3. A path
    To be implemented
4. Multiple numbers, or -1 for all files.
    To be implemented
5. Toggle sorting strategy.
    To be implemented
"""


def get_files_cli(init_dir: bytes | str | Path, multiple: bool = True,
                  dir_sort: bool = False, name_sort: bool = True) -> List[Path]:
    start_path: Path = Path(init_dir)
    print(f"\nCurrent directory: \x1B[34m{start_path}\x1B[39m")
    start_path_contents = list(start_path.iterdir())

    # Notify if empty
    if len(start_path_contents) < 1:
        print("This folder is empty.")

    # in-place sort for both display and selection
    if dir_sort and name_sort:
        _dirs = list(filter(lambda x: x.is_dir(), start_path_contents))
        _files = list(filter(lambda x: x.is_file(), start_path_contents))
        _dirs.sort()
        _files.sort()
        start_path_contents = _dirs + _files
    elif dir_sort:
        _dirs = list(filter(lambda x: x.is_dir(), start_path_contents))
        _files = list(filter(lambda x: x.is_file(), start_path_contents))
        start_path_contents = list(_dirs) + list(_files)
    elif name_sort:
        start_path_contents.sort()

    # Print with or without [Dir]
    if any(f.is_dir() for f in start_path_contents):
        dir_marker = "[Dir] "
        marker_len = len(dir_marker)
        for i, f in enumerate(start_path_contents):
            # Print only the filename
            print(
                f"{i:>3}: \x1B[31m{dir_marker if f.is_dir() else ' '*marker_len}\x1B[39m{f.name}"
            )
    else:
        for i, f in enumerate(start_path_contents):
            print(f"{i:>3}: {f.name}")

    while True:
        # Get user input
        try:
            my_input = input("Select file to parse, ? for more details: ")
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt recieved. Exiting.")
            sys.exit(0)

        # Convert user input into Path for return
        try:
            # Single selection case
            my_input = int(my_input)
            if -1 <= my_input < len(start_path_contents):
                break
            if my_input >= len(start_path_contents):
                print("Value given is too large!")
            else:
                print("Value given is too small!")
        except ValueError:
            # Possible other selection cases or menu options.
            # Regex cases
            # https://stackoverflow.com/a/72201246
            match my_input:
                # Case 1: Help Menu
                # Case 2: Go up
                # Case 3: A path is given
                # Case 4: Multiple integers that are comma/space?-seperated
                case '?' | 'm':
                    print(_HELP_TEXT)

                case '..':
                    return get_files_cli(start_path.parent.resolve(),
                                         multiple=multiple, dir_sort=dir_sort,
                                         name_sort=name_sort)

                case _:
                    raise NotImplementedError

    if my_input == -1:
        result = list(filter(lambda x: x.is_file(), start_path_contents))
    else:
        file_chosen = start_path_contents[my_input]
        if file_chosen.is_file():
            result = file_chosen if not multiple else [file_chosen,]
        elif file_chosen.is_dir():
            result = get_files_cli(
                start_path.joinpath(file_chosen),
                multiple=multiple,
                dir_sort=dir_sort,
                name_sort=name_sort
            )
        else:
            print("Selection is neither file nor directory.")
            sys.exit(0)

    return result
