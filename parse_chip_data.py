import re
import argparse


def parse_chip_data(data):
    chip_failures = {}
    for line in data.split("\n"):
        chip_match = re.match(r"Chip \((\d+), (\d+)\) failed (\d+) times", line)
        if chip_match:
            y, x, failures = map(int, chip_match.groups())
            chip_failures[(y, x)] = failures
    return chip_failures


def print_grid(filename, chip_failures):
    grid = [[0 for _ in range(4)] for _ in range(8)]
    for (y, x), failures in chip_failures.items():
        if 0 <= y < 8 and 0 <= x < 4:
            grid[y][x] = failures

    print(f"Chip failures for {filename}")
    config = re.match(r".*(False|True)_(False|True).*", filename)
    if config:
        act_sharded, weight_sharded = map(lambda x: x == "True", config.groups())
        print(f"Act sharded: {act_sharded}, Weight sharded: {weight_sharded}")
        if act_sharded and weight_sharded:
            print(f"DRAM sharded matmul")
        elif act_sharded:
            print(f"Matmul 1D")
        else:
            print(f"Matmul 2D")

    print("+--------+--------+--------+--------+")
    for row in grid:
        print("|{:^8}|{:^8}|{:^8}|{:^8}|".format(*row))
        print("+--------+--------+--------+--------+")


if __name__ == "__main__":
    # get filename from arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="filename to read data from")
    parser.add_argument("--dir", help="directory to read data from")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            data = f.read()

        chip_failures = parse_chip_data(data)
        print_grid(args.file, chip_failures)
    else:
        assert args.dir
        import os

        for filename in os.listdir(args.dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(args.dir, filename)) as f:
                data = f.read()

            chip_failures = parse_chip_data(data)
            print_grid(filename, chip_failures)
