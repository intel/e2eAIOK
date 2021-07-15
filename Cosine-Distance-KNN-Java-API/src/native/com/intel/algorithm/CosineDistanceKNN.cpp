#include "com_intel_algorithm_CosineDistanceKNN.h"
#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/common.hpp"
#include <jni.h>
#include <string>
// Just for cout table use. To find the header, need add the below option in compiling.
// -I/${ONEDAL_HOME}/__release_lnx_gnu/daal/latest/examples/oneapi/cpp/source
#include "example_util/utils.hpp"

#define JAVA_WRAPPER_CLASS "com/intel/algorithm/CosineDistanceKNN"

namespace dal = oneapi::dal;
namespace knn = dal::knn;

// TODO: free resources.
static int createTableOnJVM(const oneapi::dal::table &table, const std::string& initTableMethod,
    const std::string& setTableMethod, JNIEnv *env) {
    jclass clazz = env->FindClass(JAVA_WRAPPER_CLASS);
    if (clazz == NULL) {
      return -1;
    }
    // Call initIndicesTable
    jmethodID init_mid = env->GetStaticMethodID(clazz, initTableMethod.c_str(), "(II)V");
    env->CallStaticVoidMethod(clazz, init_mid, table.get_row_count(),
      table.get_column_count());

    jmethodID set_mid;
    // Different arg type is considered.
    if (setTableMethod == "setIndices") {
      set_mid = env->GetStaticMethodID(clazz, setTableMethod.c_str(), "(II)V");
    } else {
      set_mid = env->GetStaticMethodID(clazz, setTableMethod.c_str(), "(IF)V");
    }

    auto arr = oneapi::dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();
    for (std::int64_t i = 0; i < table.get_row_count(); i++) {
        for (std::int64_t j = 0; j < table.get_column_count(); j++) {
             // explicitly casting is necessary.
            if (setTableMethod == "setIndices") {
                env->CallStaticVoidMethod(clazz, set_mid, i * table.get_column_count() + j,
                (int)x[i * table.get_column_count() + j]);
            } else {
                env->CallStaticVoidMethod(clazz, set_mid, i * table.get_column_count() + j,
                x[i * table.get_column_count() + j]);
            }
        }
    }
    return 0;
}

/*
 * KNN brute force search based on cosine distance.
 *
 * Class:     com_intel_algorithm_CosineDistanceKNN
 * Method:    search
 * Signature: (ILjava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_com_intel_algorithm_CosineDistanceKNN_search(JNIEnv *env,
    jclass thisClass, jint neighbors_count, jstring train_data_path, jstring query_data_path) {

    const auto train_data_file_path = env->GetStringUTFChars(train_data_path, NULL);
    const auto query_data_file_path = env->GetStringUTFChars(query_data_path, NULL);

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_path });
    const auto x_query = dal::read<dal::table>(dal::csv::data_source{ query_data_file_path });

    using cosine_desc_t = dal::cosine_distance::descriptor<float>;
    const auto cosine_desc = cosine_desc_t{};

    const auto knn_desc =
        knn::descriptor<float, knn::method::brute_force, knn::task::search, cosine_desc_t>(
            neighbors_count,
            cosine_desc);

    const auto train_result = dal::train(knn_desc, x_train);
    const auto test_result = dal::infer(knn_desc, x_query, train_result.get_model());

    // #include "example_util/utils.hpp"
    std::cout << "Indices result:\n" << test_result.get_indices() << std::endl;
    std::cout << "Distance result:\n" << test_result.get_distances() << std::endl;

    const auto indices_table =  test_result.get_indices();
    const auto distances_table = test_result.get_distances();

    int res = createTableOnJVM(indices_table, "initIndicesTable", "setIndices", env);
    if (res == -1) {
      return -1;
    }
    res = createTableOnJVM(distances_table, "initDistancesTable", "setDistances", env);
    return res;
}