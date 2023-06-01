import pickle
from pathlib import Path
from workshop import *

def save_code(n_samples, directory, output_path):
    directory = Path(directory)
    all_snippets = [[] for _ in range(427)]
    for j in range(n_samples):
        path = directory / f"output_{j}"
        snippets = read_output_file(path)
            
        if len(snippets) == 428:
            assert snippets[-1] == ""
            snippets = snippets[:-1]
        else:
            assert len(snippets) == 427

        for snippet_set, one_sample in zip(all_snippets, snippets):
            snippet_set.append(one_sample)
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wb") as file:
        pickle.dump(all_snippets, file)

if __name__ == "__main__":
    output_root = output_directory("cleaning-data")

    n_runs = 30
    n_samples = 20
    root = Path("results/codeparrot-mbpp-28-04-23@20:50:35")
    for i in range(n_runs):
        directory = root / f"seed_{i}"
        output_path = output_root / "random_prefixes" / f"prefix_{i}" / "code.pickle"
        save_code(n_samples, directory, output_path)        

    save_code(
        n_samples,
        "results/10-05-23@13:02:33/bottom",
        output_path = output_root / "bert_agnostic" / "bottom.pickle"
    )

    save_code(
        n_samples,
        "results/10-05-23@13:02:33/top",
        output_root / "bert_agnostic" / "top.pickle"
    )

    save_code(
        n_samples,
        "results/23-04-23@14:33:31",
        output_root / "bert_aware" / "code.pickle"
    )

    save_code(
        n_samples,
        "results/random-python-10-05-23@20:23:17/top",
        output_root / "code_only" / "code.pickle"
    )

    save_code(
        n_samples,
        "results/mbpp-noprefix-09-05-23@18:10:30",
        output_root / "no_prefixes" / "code.pickle"
    )