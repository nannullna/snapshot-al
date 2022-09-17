import json

def write_json(obj, path: str, verbose: bool=False, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, **kwargs)
    if verbose:
        print(f"Results saved to {path}.")