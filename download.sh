#!/usr/bin/env bash
# Download the interpreter segments of all three channels (PrimaTV, ProTV,
# Digi24) in parallel, one downloader per channel, into this script's folder:
#
#     mp4s/PrimaTV   mp4s/ProTV   mp4s/Digi24   logs/
#
# The downloaders are in scripts/ and read the segment lists in manifests/.
#
# It runs them with the Python environment of requirements.txt (.venv) and checks
# what else they need (see --help). Stop it with Ctrl-C and run the same command again to resume:
# finished segments are skipped, an interrupted one is fetched again from its
# start, and the logs are appended to.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$HERE"
COOKIES=
PY="$HERE/.venv/bin/python"
YTDLP=
ONLY=primatv,protv,digi24
DRY_RUN=
SETUP=1
ENV_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/rsl-news-download"
POT_PORT=4416  # where the PO-token plugin looks for its server by default
PRIMA_WORKERS=4
PROTV_WORKERS=16
DIGI_WORKERS=8

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Downloads PrimaTV, ProTV and Digi24 at the same time, into mp4s/ next to this
script (logs in logs/). These are public servers: please be nice to them; lower
the numbers if a server starts refusing or slowing down.

Options:
  -c, --cookies FILE        Netscape cookies.txt from a logged-in YouTube
                            session; required for PrimaTV
  -o, --only LIST           comma-separated subset of primatv,protv,digi24
                            (default: all three)
  -n, --dry-run             only list what would be downloaded
      --prima-workers N     PrimaTV videos at once (default: $PRIMA_WORKERS)
      --protv-workers N     ProTV episodes at once (default: $PROTV_WORKERS)
      --digi-workers N      Digi24 days at once (default: $DIGI_WORKERS)
      --python PATH         Python with the packages of requirements.txt
                            (default: .venv/bin/python next to this script)
      --yt-dlp PATH         yt-dlp to use (default: the one next to --python)
      --env DIR             where the PO-token server is built
                            (default: $ENV_DIR)
      --no-setup            update and build nothing: use --yt-dlp as it is and
                            a PO-token server that you provide
  -h, --help                show this help

Needed (checked, not installed):
  all channels   the Python environment of requirements.txt (README.md),
                 ffmpeg and ffprobe
  PrimaTV        Node.js >= 20; git and npm on the first run only

Done for PrimaTV (unless --no-setup): yt-dlp is updated on every run (YouTube
changes often), and the bgutil PO-token server of the same version as the
plugin is built (first run only), started for the run and stopped afterwards.
One already listening on 127.0.0.1:$POT_PORT is used as it is.
EOF
}

value() {  # the option's value, or an error if there is none
    if [ $# -lt 2 ] || [ -z "$2" ]; then
        echo "option $1 needs a value" >&2
        exit 2
    fi
    printf '%s' "$2"
}

number() {  # a positive integer, or an error
    case "$2" in
        '' | *[!0-9]* | 0) echo "option $1 needs a positive integer, got '$2'" >&2; exit 2 ;;
    esac
    printf '%s' "$2"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -c | --cookies)     COOKIES=$(value "$@") || exit 2; shift 2 ;;
        -o | --only)        ONLY=$(value "$@") || exit 2; shift 2 ;;
        -n | --dry-run)     DRY_RUN=--dry-run; shift ;;
        --prima-workers)    PRIMA_WORKERS=$(number "$1" "${2:-}") || exit 2; shift 2 ;;
        --protv-workers)    PROTV_WORKERS=$(number "$1" "${2:-}") || exit 2; shift 2 ;;
        --digi-workers)     DIGI_WORKERS=$(number "$1" "${2:-}") || exit 2; shift 2 ;;
        --env)              ENV_DIR=$(value "$@") || exit 2; shift 2 ;;
        --no-setup)         SETUP=; shift ;;
        --python)           PY=$(value "$@") || exit 2; shift 2 ;;
        --yt-dlp)           YTDLP=$(value "$@") || exit 2; shift 2 ;;
        -h | --help)        usage; exit 0 ;;
        *)                  echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

wanted() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

for channel in ${ONLY//,/ }; do
    case "$channel" in
        primatv | protv | digi24) ;;
        *) echo "unknown channel in --only: $channel" >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }
say() { echo "[setup] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# PrimaTV needs yt-dlp, node, cookies and the PO-token server only to download.
FETCH_PRIMA=
wanted primatv && [ -z "$DRY_RUN" ] && FETCH_PRIMA=1

# ---- what must already be on the system -----------------------------------
missing=()
have ffmpeg || missing+=(ffmpeg)
have ffprobe || missing+=("ffprobe (part of ffmpeg)")
[ -n "$FETCH_PRIMA" ] && ! have node && missing+=("Node.js >= 20")
[ ${#missing[@]} -eq 0 ] || die "not found: ${missing[*]}. Install them and run again."

[ -x "$PY" ] || die "no Python environment at $PY; create it (README.md):
       python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt"
"$PY" -c 'import requests, bs4' 2>/dev/null \
    || die "$PY cannot import requests and bs4: install requirements.txt into it"
if [ -n "$FETCH_PRIMA" ]; then
    node -e 'process.exit(parseInt(process.versions.node) >= 20 ? 0 : 1)' \
        || die "Node.js >= 20 is needed (found $(node --version))"
    [ -n "$COOKIES" ] || die "PrimaTV needs --cookies FILE (a Netscape cookies.txt exported from a
       logged-in YouTube session), or leave it out with --only protv,digi24"
    [ -r "$COOKIES" ] || die "cannot read the cookies file $COOKIES"
fi

mkdir -p "$ROOT/logs" || die "cannot create $ROOT"

# ---- yt-dlp (PrimaTV) ------------------------------------------------------
[ -n "$YTDLP" ] || YTDLP="$(dirname "$PY")/yt-dlp"
if [ -n "$FETCH_PRIMA" ]; then
    if [ -n "$SETUP" ]; then
        say "updating yt-dlp"
        "$PY" -m pip install --quiet --disable-pip-version-check --upgrade yt-dlp \
            || die "pip install --upgrade yt-dlp failed"
    fi
    have "$YTDLP" || die "yt-dlp not found ($YTDLP)"
fi

# Ctrl-C / kill stops every downloader and its ffmpeg / yt-dlp children.
trap 'trap - INT TERM; kill 0' INT TERM

# ---- PO-token server (PrimaTV) ---------------------------------------------
POT_PID=
trap 'if [ -n "$POT_PID" ]; then kill "$POT_PID" 2>/dev/null; wait "$POT_PID" 2>/dev/null; fi' EXIT
if [ -n "$FETCH_PRIMA" ]; then
    if port_open "$POT_PORT"; then
        say "using the PO-token server already listening on 127.0.0.1:$POT_PORT"
    elif [ -z "$SETUP" ]; then
        die "--no-setup: no PO-token server on 127.0.0.1:$POT_PORT (see scripts/download_prima.py --help)"
    else
        POT_VERSION=$("$PY" -c 'import importlib.metadata as m; print(m.version("bgutil-ytdlp-pot-provider"))' 2>/dev/null) \
            || die "$PY has no bgutil-ytdlp-pot-provider: install requirements.txt into it"
        repo="$ENV_DIR/bgutil-ytdlp-pot-provider-$POT_VERSION"
        if [ ! -f "$repo/server/build/main.js" ]; then
            have git || die "git is needed (once) to fetch the PO-token server"
            have npm || die "npm is needed (once) to build the PO-token server"
            say "fetching and building the PO-token server $POT_VERSION (first run only)"
            rm -rf "$repo"  # left over from an interrupted first run
            mkdir -p "$ENV_DIR" || die "cannot create $ENV_DIR"
            git clone --quiet --depth 1 --branch "$POT_VERSION" \
                https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "$repo" \
                || die "git clone of the PO-token server failed"
            (cd "$repo/server" && npm ci --no-audit --no-fund --loglevel=error && npx --no-install tsc) \
                || die "building the PO-token server failed"
        fi
        say "starting the PO-token server (log: $ROOT/logs/pot-server.log)"
        (cd "$repo/server" && exec node build/main.js --port "$POT_PORT") >>"$ROOT/logs/pot-server.log" 2>&1 &
        POT_PID=$!
        for _ in $(seq 60); do
            port_open "$POT_PORT" && break
            kill -0 "$POT_PID" 2>/dev/null || break
            sleep 1
        done
        port_open "$POT_PORT" || die "the PO-token server did not start, see $ROOT/logs/pot-server.log"
    fi
fi

# ---- downloads ---------------------------------------------------------------
PIDS=()
NAMES=()
run() {  # run <name> <command...>: in the background, log to <root>/logs/<name>.log
    local name=$1
    shift
    echo "===== run started $(date '+%Y-%m-%d %H:%M:%S') =====" >>"$ROOT/logs/$name.log"
    (
        set -o pipefail
        "$@" 2>&1 | tee -a "$ROOT/logs/$name.log" | sed -u "s/^/[$name] /"
    ) &
    PIDS+=("$!")
    NAMES+=("$name")
}

if wanted primatv; then
    prima=("$PY" -u "$HERE/scripts/download_prima.py" --root "$ROOT/mp4s/PrimaTV" --yt-dlp "$YTDLP"
           --workers "$PRIMA_WORKERS")
    [ -n "$COOKIES" ] && prima+=(--cookies "$COOKIES")
    [ -n "$DRY_RUN" ] && prima+=("$DRY_RUN")
    run PrimaTV "${prima[@]}"
fi
if wanted protv; then
    protv=("$PY" -u "$HERE/scripts/download_protv.py" --root "$ROOT/mp4s/ProTV" --workers "$PROTV_WORKERS")
    [ -n "$DRY_RUN" ] && protv+=("$DRY_RUN")
    run ProTV "${protv[@]}"
fi
if wanted digi24; then
    digi=("$PY" -u "$HERE/scripts/download_digi24.py" --root "$ROOT/mp4s/Digi24" --workers "$DIGI_WORKERS")
    [ -n "$DRY_RUN" ] && digi+=("$DRY_RUN")
    run Digi24 "${digi[@]}"
fi

status=0
for n in "${!PIDS[@]}"; do
    if wait "${PIDS[$n]}"; then
        echo "[${NAMES[$n]}] finished"
    else
        echo "[${NAMES[$n]}] FAILED or incomplete, see $ROOT/logs/${NAMES[$n]}.log"
        status=1
    fi
done
exit $status
