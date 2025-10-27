#!/usr/bin/env bash
set -e

: ${ADLER32:=$(command -v xrdadler32)}
: ${CRC32C:=$(command -v xrdcrc32c)}
: ${XRDCP:=$(command -v xrdcp)}
: ${XRDFS:=$(command -v xrdfs)}
: ${OPENSSL:=$(command -v openssl)}
: ${CURL:=$(command -v curl)}

# Parallel arrays for compatibility with Bash < 4
host_names=(metaman man1 man2 srv1 srv2 srv3 srv6)
host_ports=(10970 10971 10972 10973 10974 10975 10976)
host_roots=()
host_https=()

# Populate derived URLs
for ((i = 0; i < ${#host_names[@]}; i++)); do
    port="${host_ports[i]}"
    root_url="root://localhost:$port"
    http_url="${root_url/root/http}"
    host_roots+=("$root_url")
    host_https+=("$http_url")
done

get_index_for_host() {
    local host="$1"
    for ((i = 0; i < ${#host_names[@]}; i++)); do
        if [[ "${host_names[i]}" == "$host" ]]; then
            echo "$i"
            return
        fi
    done
    echo "-1"
}

get_http_url_for_host() {
    local host="$1"
    local idx
    idx=$(get_index_for_host "$host")
    if [[ "$idx" -ge 0 ]]; then
        echo "${host_https[idx]}"
    else
        echo "Unknown host: $host" >&2
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

check_required_commands() {
    for prog in "$@"; do
        if [[ ! -x "$prog" ]]; then
            echo >&2 "$(basename "$0"): error: '$prog': command not found"
            exit 1
        fi
    done
}

# shellcheck disable=SC2329
cleanup() {
    echo "Error occurred. Cleaning up..."
}
trap "cleanup; exit 1" ABRT

setup_scitokens() {
    local issuer_dir="$XRDSCITOKENS_ISSUER_DIR"
    local token_file="${PWD}/generated_tokens/token"

    if ! "$XRDSCITOKENS_CREATE_TOKEN" \
        "$issuer_dir/issuer_pub_1.pem" \
        "$issuer_dir/issuer_key_1.pem" \
        test_1 \
        "https://localhost:7095/issuer/one" \
        "storage.modify:/ storage.create:/ storage.read:/" \
        1800 >"$token_file"; then
        echo "Failed to create token"
        exit 1
    fi
    chmod 0600 "$token_file"
}

upload_file_to_host() {
    local host="$1"
    local remote_name="$2"
    local file_path="$3"

    local http_url
    http_url=$(get_http_url_for_host "$host")

    echo -e "\nUploading '$file_path' to '${http_url}/$RMTDATADIR/$remote_name'"
    if ! ${CURL} -L -s -H "Authorization: Bearer ${BEARER_TOKEN}" \
        "${http_url}/$RMTDATADIR/$remote_name" -T "$file_path"; then
        echo "Upload to $host failed!"
        exit 1
    fi
}

perform_rename() {
    local src_host="$1"
    local old_name="$2"
    local new_name="$3"
    local expected_code="$4"

    local http_url
    http_url=$(get_http_url_for_host "$src_host")

    local src_url="${http_url}/$RMTDATADIR/$old_name"
    local dst_url="${http_url}/$RMTDATADIR/$new_name"

    echo -e "\nRenaming '$src_url' to '$dst_url'"

    local tmp_body
    local tmp_stderr
    tmp_body=$(mktemp /tmp/rename_curl_body_XXXX)
    tmp_stderr=$(mktemp /tmp/rename_curl_stderr_XXXX)

    local http_code
    http_code=$(${CURL} -s -S -L -X MOVE -o "$tmp_body" -w "%{http_code}" -v \
        -H "Authorization: Bearer ${BEARER_TOKEN}" \
        -H "Destination: $dst_url" \
        "$src_url" 2>"$tmp_stderr")

    if [[ "$http_code" != "$expected_code" ]]; then
        echo "Rename failed from '$old_name' to '$new_name': expected $expected_code, got $http_code"
        echo "----- CURL VERBOSE LOG -----"
        cat "$tmp_stderr"
        echo "----- CURL RESPONSE BODY -----"
        cat "$tmp_body"
        echo "------------------------------"
        rm -f "$tmp_body" "$tmp_stderr"
        exit 1
    fi

    rm -f "$tmp_body" "$tmp_stderr"
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

check_required_commands "$ADLER32" "$CRC32C" "$XRDCP" "$XRDFS" "$OPENSSL" "$CURL"

RMTDATADIR="/srvdata"
LCLDATADIR="${PWD}/localdata"
mkdir -p "$LCLDATADIR" "$PWD/generated_tokens"
mkdir -p "$XDG_CACHE_HOME/scitokens" && rm -rf "$XDG_CACHE_HOME/scitokens"/*

setup_scitokens
export BEARER_TOKEN_FILE="${PWD}/generated_tokens/token"
BEARER_TOKEN=$(cat "$BEARER_TOKEN_FILE")
export BEARER_TOKEN

# Create random file
testfile="${LCLDATADIR}/randomfile.ref"
${OPENSSL} rand -out "$testfile" $((1024 * (RANDOM + 1)))

# The copy of cgi parameters to destination ensure that the move suceeds
src_hosts=(metaman man1 srv1)
src_codes=(501 201 201)

for ((i = 0; i < ${#src_hosts[@]}; i++)); do
    src="${src_hosts[i]}"
    upload_file_to_host "$src" "old_$src" "$testfile"
done

for ((i = 0; i < ${#src_hosts[@]}; i++)); do
    src="${src_hosts[i]}"
    code="${src_codes[i]}"
    perform_rename "$src" "old_$src" "new_$src" "$code"
done

echo -e "\nALL TESTS PASSED"
exit 0
