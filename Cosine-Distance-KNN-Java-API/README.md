## Java Wrapper for Cosine Distance Based KNN Algorithm

### Introduction

This project provides a Java wrapper for cosine distance based KNN algorithm of oneDAL.

In current implementation, the program (C++ part) will read train/test data from two CSV files. And the paths for them are
required to be passed through Java API. User can also provide neighbors count through Java API. As the result of inference,
two tables will be created in JVM: indices table & distances table.

### Requirements

* JDK
* gcc/g++ (6.x or above, 7.3.1 is workable)
* Intel OneDAL

### Build OneDAL (Optional)

You can directly install oneDAL libs if a released version meets your requirement.

In currently latest release, cosine distance based KNN algorithm is not included. So we need build oneDAL from source code.

Github link: https://github.com/oneapi-src/oneDAL

* For resolving JNI dependencies (can be skipped if already set):

  `export JAVA_HOME=<YOUR_JDK_HOME>`

  `export PATH=$JAVA_HOME/bin:$PATH`

  `export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/linux:$CPATH`

* Download dependencies:

  Go to OneDAL source code home directory.

  `./dev/download_micromkl.sh`

  `./dev/download_tbb.sh`

  `source ./__deps/tbb/lnx/env/vars.sh intel64`

* Build oneDAL:

  `make -f makefile daal oneapi_c PLAT=lnx32e COMPILER=gnu`

  `source <ONEDAL_HOME>/__release_lnx_gnu/daal/latest/env/vars.sh`

  The above `source` command will set LD_LIBRARY_PATH/LIBRARY_PATH to include `<ONEDAL_HOME>/__release_lnx_gnu/daal/latest/lib/intel64`

### Build the current project

* Build Java wrapper and generate header

  Navigate to `<PROJECT_HOME>/src/java`.

  `javac -h . com/intel/algorithm/CosineDistanceKNN.java`

  Then, `com_intel_algorithm_CosineDistanceKNN.h` will be generated in current path.

* Build C++ code

  `g++ ../native/com/intel/algorithm/CosineDistanceKNN.cpp -L<ONEDAL_HOME>/__release_lnx_gnu/daal/latest/lib/intel64 \
  -lonedal_core -lonedal -lonedal_sequential -lonedal_thread -lJavaAPI -std=c++17 -fPIC -shared -o libknn.so -I$JAVA_HOME/include \
  -I$JAVA_HOME/include/linux -I<PROJECT_HOME>/src/java -I<ONEDAL_HOME>/__release_lnx_gnu/daal/latest/include \
  -I<ONEDAL_HOME>/__release_lnx_gnu/daal/latest/examples/oneapi/cpp/source`

  Note:

  `-L<ONEDAL_HOME>/__release_lnx_gnu/daal/latest/lib/intel64` can be removed if the path is already included in `LD_LIBRARY_PATH`.

  `-I$JAVA_HOME/include` & `-I$JAVA_HOME/include/linux` should be added to help the compiler find `jni.h`, provided by JDK.

  `-I<PROJECT_HOME>/src/java` provides the path where `com_intel_algorithm_CosineDistanceKNN.h` is located.

  `-I<ONEDAL_HOME>/__release_lnx_gnu/daal/latest/include` provides the path where program required headers are located. Headers inside sub-dir
  can be used by program via `#include "sub-dir/xxx.h"`.

  `-I<ONEDAL_HOME>/__release_lnx_gnu/daal/latest/examples/oneapi/cpp/source` is required since `#include "example_util/utils.hpp"` is introduced
  into our C++ program to facilitate printing result.

  A shared lib named `libknn.so` will be created in current directory. The lib name is hard coded in Java code. So do NOT change it.

  You can use `ldd -r libknn.so` to check whether there is any "undefined symbol" issue, which is generally caused by incorrectly setting for
  `-L` (or `LD_LIBRARY_PATH`) or `-l` in compiling. 

### Test

* Train/test sample data

  `<ONEDAL_HOME>/examples/oneapi/data/k_nearest_neighbors_train_data.csv`

  `<ONEDAL_HOME>/examples/oneapi/data/k_nearest_neighbors_test_data.csv`

* Compile/run test code

  Navigate to `<PROJECT_HOME>/src/java`.

  `javac com/intel/algorithm/Main.java`

  `java -Djava.library.path=. com.intel.algorithm.Main <PATH_TO_TRAIN_DATA> <PATH_TO_TEST_DATA> > output.log`

  The path for shared lib, `libknn.so`, is specified via `-Djava.library.path`.

### Integration to Your Project

* Install OneDAL (for future release contains cosine distance based KNN) or build from source code.

* Introduce the below source code into your project

  `./src/java/com/intel/algorithm/CosineDistanceKNN.java` (Java API)

  `./src/java/com/intel/algorithm/Table.java`

  `./src/native/com/intel/algorithm/CosineDistanceKNN.cpp`

  You need to build a shared lib named `libknn.so`, as the above instructions show.

### Future Work

The current implementation is quite straightforward. In the future, we can consider the seamless integration into frameworks
or pipelines. If test data comes from other source of Java program, we will need to pass the data to C++ code through JNI.

Another consideration is data scale. If there is very large amount of data as result, creating tables on JVM will consume a lot
of memory resource and bring non-ignorable latency. In this case, it may be better to keep the data on native memory and
provide Java API for accessing certain row(s) of data.

## Reference:

* https://github.com/oneapi-src/oneDAL/blob/master/INSTALL.md
* http://oneapi-src.github.io/oneDAL/onedal/get-started.html#onedal-get-started