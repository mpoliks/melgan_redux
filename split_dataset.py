from pathlib import Path
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--shuffle", type=bool, default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    supported_extensions = set([".wav", ".aif", ".aiff", ".mp3", ".m4a"])
    audio_paths = list(Path(args.data_path).glob("*.*"))
    # check files in filepath_list is supported (by extensions)
    audio_paths = [
        path
        for path in audio_paths
        if (
            str(Path(path).suffix).lower() in supported_extensions
            and not str(path).startswith("__MACOSX")
        )
    ]
    num_files = len(audio_paths)

    test_fraction = 0.1
    if args.shuffle:
        np.random.seed(123)
        audio_paths = np.random.permutation(audio_paths)
    else:
        audio_paths = sorted(audio_paths)
    split_at = int((1 - test_fraction) * num_files)
    train_paths = audio_paths[:split_at]
    test_paths = audio_paths[split_at:]

    with open("train_files.txt", "w") as f:
        for path in sorted(train_paths):
            print(path, file=f)

    with open("test_files.txt", "w") as f:
        for path in sorted(test_paths):
            print(path, file=f)


if __name__ == "__main__":
    main()
