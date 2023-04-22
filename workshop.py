import datetime
import os
from pathlib import Path
import sys


def output_directory(name):
    output_home = Path("../results")
    now = datetime.now().strftime("%d-%m-%y@%H:%M:%S")
    output_directory_name = f"{name}-{now}"
    output_directory = output_home / output_directory_name
    output_directory.mkdir(parents=True, exist_ok=True)
    
    dump_details(output_directory)
    
    return output_directory


def dump_details(output_directory):
    root = Path(os.environ["CODEROOT"]).resolve()
    all_modules = set()
    for name in sys.modules:
        module = sys.modules[name]
        try:
            module_file = Path(module.__file__)
            is_descendant = root in module_file.resolve().parents
            if is_descendant:
                all_modules.add(module_file.resolve())
        except (AttributeError, TypeError):
            pass

    Path(output_directory / "details").mkdir(exist_ok=True, parents=True)

    code_directory = Path(output_directory) /  "details" / "code"
    for module_path in all_modules:
        target_path = code_directory / module_path.relative_to(root)
        target_path.parent.mkdir(exist_ok=True, parents=True)
        with open(target_path, "w") as file, open(module_path, "r") as source_file:
            file.write(source_file.read())

    with open(Path(output_directory) / "details"  / "run.sh", "w") as file:
        print(*sys.argv, file=file)