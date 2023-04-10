git clone https://github.com/intel/e2eAIOK.git; cd e2eAIOK; git submodule update --init --recursive
mv e2eAIOK e2eaiok
cd e2eaiok/modelzoo/dien/train; sh patch_dien.sh