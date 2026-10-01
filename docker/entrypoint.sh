#!/bin/sh
# entrypoint.sh -- repair state-directory ownership, then drop privileges.
#
# The container runs as the non-root `agent` user (uid 1000), but the two
# directories it must write to can arrive owned by somebody else:
#
#   * /app/data -- Docker seeds a named volume from the image exactly ONCE.
#     A volume first created by an older image (one with no /app/data, or with
#     a root-owned one) is never re-seeded, so it stays wrong forever.
#     `compose down` keeps it; only `down -v` removes it.
#   * /app/workspace -- a bind mount, owned by whatever uid the host uses.
#
# Either way ChromaDB fails with "Permission denied (os error 13)" before the
# sample receipts are ever copied, and it looks like two unrelated bugs. Doing
# this at startup means a stale volume heals itself on the next run instead of
# requiring the user to know the magic incantation.
#
# Ownership is only rewritten when it is actually broken. A blanket chown -R on
# every start would reach real files inside the bind-mounted workspace and
# change their owner on a native Linux host -- a surprising side effect for
# something nobody asked for.

set -eu

fix_state_dir() {
    dir="$1"
    mkdir -p "$dir" 2>/dev/null || true

    # `gosu agent test -w` answers "could the non-root user write here?"
    # without us having to know its uid, and without a root-owned file
    # anywhere else in the tree getting touched.
    if gosu agent test -w "$dir" 2>/dev/null; then
        return 0
    fi

    echo "entrypoint: '$dir' is not writable by 'agent' -- repairing ownership" >&2
    if ! chown -R agent:agent "$dir"; then
        echo "entrypoint: could not chown '$dir'. It may be mounted read-only," >&2
        echo "entrypoint: or the volume may be in a state this image cannot fix." >&2
        echo "entrypoint: Try: docker compose down -v   (then start again)" >&2
        exit 1
    fi
}

fix_state_dir /app/data
fix_state_dir /app/workspace

# Hand over to the non-root user for the real command. Nothing in the
# application ever runs as root.
exec gosu agent "$@"
