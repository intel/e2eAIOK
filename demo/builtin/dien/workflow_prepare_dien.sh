git clone https://github.com/intel/e2eAIOK.git;
mv e2eAIOK e2eaiok
cd e2eaiok; git submodule update --init --recursive
cd modelzoo/dien/train; sh patch_dien.sh
