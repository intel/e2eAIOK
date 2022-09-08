get_original_model () {

   cp -r ../third_party/dlrm ./
   cd dlrm

}


apply_patch () {
   patch -p1 <../dlrm.patch

}


get_original_model

apply_patch
