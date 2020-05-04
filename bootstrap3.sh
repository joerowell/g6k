#!/usr/bin/env bash


jobs="-j 4 "
if [ "$1" = "-j" ]; then
   jobs="-j $2 "
fi

# Create Virtual Environment
alias python=/usr/bin/python3
alias pip=/usr/bin/pip3

rm -rf g6k-env
virtualenv g6k-env
cat <<EOF >>g6k-env/bin/activate
### LD_LIBRARY_HACK
_OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
### END_LD_LIBRARY_HACK

### PKG_CONFIG_HACK
_OLD_PKG_CONFIG_PATH="\$PKG_CONFIG_PATH"
PKG_CONFIG_PATH="\$VIRTUAL_ENV/lib/pkgconfig:\$PKG_CONFIG_PATH"
export PKG_CONFIG_PATH
### END_PKG_CONFIG_HACK
EOF

ln -s g6k-env/bin/activate ./
source ./activate

pip install -U pip
pip install Cython
pip install cysignals


cat <<EOF >>g6k-env/bin/activate
CFLAGS="\$CFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
CXXFLAGS="\$CXXFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
export CFLAGS
export CXXFLAGS
EOF

source ./activate


# Install FPLLL

git clone https://github.com/fplll/fplll g6k-fplll
cd g6k-fplll || exit
./autogen.sh
./configure --prefix="$VIRTUAL_ENV" $CONFIGURE_FLAGS
make clean
make $jobs
make install
cd ..

# Install FPyLLL
git clone https://github.com/fplll/fpylll g6k-fpylll
cd g6k-fpylll || exit
pip install Cython
pip install -r requirements.txt
pip install -r suggestions.txt
python setup.py clean
python setup.py build_ext $jobs
python setup.py install
cd ..

pip install -r requirements.txt
python setup.py clean
python setup.py build_ext $jobs --inplace

echo "Don't forget to activate environment each time:"
echo " source ./activate"