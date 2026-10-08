# Keep pytest out of in-tree CMake build directories (fetched dependencies ship their own tests).
collect_ignore_glob = ["build*"]
