# clone repo
git clone https://github.com/intel/e2eAIOK.git
mv e2eAIOK e2eaiok
cd e2eaiok
git submodule update --init --recursive

# apply patch
cd modelzoo/WnD/TensorFlow2 && bash patch_wnd.sh