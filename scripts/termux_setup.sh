#!/data/data/com.termux/files/usr/bin/bash
# Set up the PC/Termux split deployment on Android.
#
# Heavy native packages come from Termux's own repo (pkg), not PyPI: numpy,
# opencv-python and onnxruntime publish no Android wheels, so pip would fall
# back to source builds and shadow the pkg-managed copies. That is also why
# pyproject.toml caps opencv-python to <5 -- pkg decides the phone's major and
# the PC side is aligned to match.
#
# `pkg` cannot pin versions: each package has a single version in the repo and
# downgrading is unsupported, so superseded releases are deleted rather than
# archived. This script pins what pip installs and only reports what pkg owns.
#
# --venv keeps everything pip installs inside .venv instead of the system
# site-packages. That makes cleanup easy (`rm -rf .venv`), since pip has no
# autoremove and dropping the server orphans ~14 transitive packages. It has to
# be a --system-site-packages venv: numpy, opencv-python and onnxruntime come
# from `pkg`, so a hermetic venv cannot import them.
#
# Switching modes? The environment you leave keeps its packages, so clean up
# the other side:
#   rm -rf .venv                                          # leaving the venv
#   pip uninstall -y fastapi uvicorn pyyaml edge-surveillance  # leaving system
set -euo pipefail

TUR=https://termux-user-repository.github.io/pypi/

# `pip install -e .` and `--group` both resolve paths from the current
# directory, so run from the repository root regardless of the caller's cwd.
cd "$(dirname "$0")/.."

usage() {
    cat <<EOF
Usage: ${0##*/} [--headless] [--venv]

  (no args)   pipeline + dashboard server, installed system-wide
  --headless  pipeline only; skips fastapi and uvicorn
  --venv      install pip packages into .venv instead of the system env
  -h, --help  show this message

PC equivalents:  uv sync  /  uv sync --no-group dev
EOF
}

WITH_SERVER=1
WITH_VENV=0
for arg in "$@"; do
    case "$arg" in
        --headless) WITH_SERVER=0 ;;
        --venv)    WITH_VENV=1 ;;
        -h|--help) usage; exit 0 ;;
        *)         echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

# x11-repo must come first: opencv-python is published in termux-x11, not
# termux-main, so without this repo `pkg` cannot find it at all.
pkg install -y x11-repo
pkg install -y python python-pip python-numpy opencv-python python-onnxruntime dbus

if [ "$WITH_VENV" -eq 1 ]; then
    python -m venv --system-site-packages .venv
    if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
        echo "no working pip in .venv; try: pkg install python-ensurepip-wheels" >&2
        exit 1
    fi
fi

# Always invoke pip through the target interpreter rather than as bare `pip`.
# $PREFIX/bin/pip stays on PATH after activating a venv, so a bare `pip` can
# silently act on the system environment instead of .venv.
if [ "$WITH_VENV" -eq 1 ]; then
    pip_run() { .venv/bin/python -m pip "$@"; }
else
    pip_run() { python -m pip "$@"; }
fi

pip_run install --upgrade pip

# --group requires pip >= 25.1. Termux currently ships well past that; this is
# a guard for older images.
# Capture the help text instead of piping it into `grep -q`. A pipeline would
# conflate two non-zero exits: 1 for "no --group" (feature absent) and 141 for
# the help command being killed (SIGPIPE, since grep -q exits on first match
# and pipefail promotes the producer's death). Under `set -e` those are
# indistinguishable, so a transient failure would read as "pip is too old".
# pip's help is ~15 KB and fits the pipe buffer, so SIGPIPE is not expected in
# practice; capturing keeps the two outcomes separate regardless.
pip_help="$(pip_run install --help 2>/dev/null || true)"
case "$pip_help" in
    *--group*) ;;
    *)
        echo "pip >= 25.1 is required for --group; got: $(pip_run --version)" >&2
        exit 1
        ;;
esac

pip_run install "pyyaml>=6.0.3,<7"

if [ "$WITH_SERVER" -eq 1 ]; then
    # pydantic-core is a Rust extension with no PyPI Android wheel; the Termux
    # User Repository carries a prebuilt one, so pin it to a binary rather than
    # paying for a source build (needs rust + binutils and setting `ANDROID_API_LEVEL`).
    pip_run install --group server \
        --only-binary pydantic-core \
        --extra-index-url "$TUR"
fi

# The base dependencies are the pkg-installed ones from above.
pip_run install -e . --no-deps

echo

if [ "$WITH_VENV" -eq 1 ]; then
    echo "pip-installed (in .venv):"
else
    echo "pip-installed (system):"
fi
pip_run list 2>/dev/null | grep -Ei '^(fastapi|uvicorn|pyyaml|edge-surveillance) ' || true

echo
echo "pkg-managed (not pinnable; 'pkg upgrade' moves these):"
pkg list-installed 2>/dev/null | grep -Ei 'opencv|python-numpy|python-onnxruntime|dbus' || true

# pyproject.toml pins the PC to opencv <5 to match the phone, so a Termux bump
# to 5.x is the signal to revisit that cap.
if [ "$(pkg list-installed 2>/dev/null | grep -cE '^opencv-python/4' || true)" -eq 0 ]; then
    echo "WARNING: Termux opencv-python is not 4.x." >&2
    echo "         Revisit the opencv-python <5 cap in pyproject.toml." >&2
fi