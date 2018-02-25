
fail () {
    ERROR="$?"
    echo >&2 -e "Failure: $@"
    exit "$ERROR"
}