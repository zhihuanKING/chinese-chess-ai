# cpp/ — Xiangqi rules kernel + Alpha-Beta engine

C++17 core for the Chinese-Chess (象棋) AI project. Implements the cross-language
contract in [`../INTERFACES.md`](../INTERFACES.md) (§1 coordinates/encoding,
§2 move integers `move = from*90 + to`, §3 pybind `Position` API, §8 Alpha-Beta).

## Layout

```
include/xq/   types.h  zobrist.h  position.h  search.h     (public headers)
src/          position.cpp  eval.cpp  search.cpp           (rules + engine)
              perft_main.cpp  xqab_main.cpp                (executables)
bindings/     module.cpp                                   (pybind11 -> _xqcore)
tests/        test_perft.cpp  test_rules.cpp               (unit tests)
third_party/  doctest.h                                    (mini test framework)
```

`xqcore` is a single static library reused by the Python module, the
executables and the tests, so movegen / rules / eval have exactly one source.

## Build everything with CMake + Ninja

From a build directory:

```bash
cd cpp
cmake -GNinja -B build -S .
ninja -C build
```

This produces:

- `build/perft`  — command-line perft tool
- `build/xqab`   — Alpha-Beta engine (UCCI-subset CLI)
- `build/_xqcore*.so` — Python extension (only if `pybind11` is importable;
  otherwise it is skipped with a status message and the rest still builds)
- `build/test_perft`, `build/test_rules` — test binaries

pybind11 is located automatically via `python -m pybind11 --cmakedir`
(install with `pip install pybind11`). Optimization flags `-O3 -march=native`
are applied in `Release` (the default). Disable native tuning with
`-DXQ_NATIVE=OFF`.

The Python wheel (`import xqai._xqcore`) is built by the repo-root
`pyproject.toml` via scikit-build-core, which points `cmake.source-dir` here.

## Build a single target by hand (no CMake)

```bash
g++ -std=c++17 -O3 -march=native -Iinclude \
    src/position.cpp src/eval.cpp src/search.cpp src/perft_main.cpp -o perft

g++ -std=c++17 -O3 -march=native -Iinclude \
    src/position.cpp src/eval.cpp src/search.cpp src/xqab_main.cpp -o xqab
```

## Run the tests

```bash
ninja -C build
ctest --test-dir build --output-on-failure
# or run the binaries directly:
./build/test_perft
./build/test_rules
```

Or without CMake:

```bash
g++ -std=c++17 -O2 -Iinclude -Ithird_party \
    src/position.cpp src/eval.cpp src/search.cpp tests/test_perft.cpp -o test_perft && ./test_perft
g++ -std=c++17 -O2 -Iinclude -Ithird_party \
    src/position.cpp src/eval.cpp src/search.cpp tests/test_rules.cpp -o test_rules && ./test_rules
```

## Using `perft`

```bash
./perft 4                 # start position, depth 4  -> 3290240
./perft 3 "<FEN>"         # custom FEN
./perft divide 2          # per-root-move breakdown
```

Verified reference counts for the standard start position:
`perft(1)=44, perft(2)=1920, perft(3)=79666, perft(4)=3290240`.

## Using `xqab` (Alpha-Beta engine)

One-shot best move with a millisecond budget:

```bash
./xqab --bestmove 1000 "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
# -> move=<int>  iccs=h2g2
```

Interactive UCCI subset (one command per line on stdin):

```
ucci
isready
position startpos [moves <iccs> ...]
position fen <FEN> [moves <iccs> ...]
go time <ms> | go movetime <ms> | go depth <d>
quit
```

Moves on the CLI use ICCS-style coordinates: file `a`–`i` (col 0–8) + rank
`0`–`9` counted from the **bottom (red) edge**, so internal `sq` ↔ ICCS rank is
`rank = 9 - row`. The pybind API uses the raw integer encoding `from*90+to`.

## FEN conventions

- 10 rows top→bottom (row 0 = black back rank, row 9 = red back rank), 9 files.
- Piece letters: `K A E H R C P` (uppercase = red, lowercase = black).
  `B` (bishop) and `N` (knight) are **accepted on input** as aliases for
  `E`/`H` and normalized to `E`/`H` on output.
- Side field `w` = red to move, `b` = black. Round-trip is lossless for the
  board + side + halfmove fields this kernel tracks.
