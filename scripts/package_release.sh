#!/usr/bin/env bash
# usage: scripts/package_release.sh <build dir> <package name>, the folder lands inside the build dir
set -euo pipefail

build="$1"
name="$2"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$build/$name"

ext=""
lib="libbreeze.so"
if [ -f "$build/breeze-cli.exe" ]; then
    ext=".exe"
    lib="libbreeze.dll"
fi

rm -rf "$out"
mkdir -p "$out"
for app in breeze-cli breeze-server breeze-convert breeze-quantize; do
    cp "$build/$app$ext" "$out/"
done
cp "$build/$lib" "$out/"
[ -f "$build/libbreeze.dll.a" ] && cp "$build/libbreeze.dll.a" "$out/"
cp -r "$root/include" "$out/"
cp "$root/README.md" "$root/LICENSE" "$out/"
mkdir -p "$out/docs"
cp "$root"/docs/*.md "$out/docs/"

# ggml and httplib are mit and want their notice shipped, shine is lgpl and wants its full text
notices="$out/THIRD_PARTY_NOTICES.txt"
{
    echo "Breeze-TTS-2.cpp bundles the following third party code."
    echo "Source for all of it is at https://github.com/HoppouAI/Breeze-TTS-2.cpp and its submodules."
    echo
    echo "==== ggml (https://github.com/ggml-org/ggml) ===="
    cat "$root/third_party/ggml/LICENSE"
    echo
    echo "==== cpp-httplib (https://github.com/yhirose/cpp-httplib) ===="
    cat <<'EOF'
MIT License

Copyright (c) 2024 Yuji Hirose

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
    echo
    if [ -f "$root/third_party/shine/COPYING" ]; then
        echo "==== shine mp3 encoder (https://github.com/toots/shine) ===="
        echo "Statically linked into breeze-server. To relink against a modified shine, rebuild from source."
        echo
        cat "$root/third_party/shine/COPYING"
    fi
} > "$notices"

echo "packaged $out"
