from nnsight import LanguageModel

from pathlib import Path


def get_dataset_path(model: LanguageModel) -> Path:
    path_parts = model.config.name_or_path.split("/")
    if len(path_parts) == 2:
        path_parts[0] = "data"
    else:
        path_parts.insert(0, "data")

    return Path(*path_parts)
