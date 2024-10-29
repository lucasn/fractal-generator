make -C doc clean
make -C doc html
python3 -m http.server -d ./doc/build/html