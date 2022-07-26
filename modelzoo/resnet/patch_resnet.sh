get_original_model () {

   cp -r ../third_party/mlperf_v1.0/Intel/benchmarks/resnet/2-nodes-16s-8376H-tensorflow/* ./

}



apply_patch () {
   patch -p5 < resnet.patch

}

get_original_model

apply_patch