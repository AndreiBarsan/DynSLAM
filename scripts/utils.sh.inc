
fail () {
    ERROR="$?"
    echo >&2 "Failure: $@"
    exit "$ERROR"
}