get_original_model () {
   cp -r ../third_party/dlrm ./
   cd dlrm
   git checkout 5634274824bab6843e82d64dad6d728232fc0354

}


apply_patch () {
   patch -p1 <../dlrm.patch

}


get_original_model

apply_patch
