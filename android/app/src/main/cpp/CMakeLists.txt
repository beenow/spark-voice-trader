# The JNI shim only. llama.cpp itself is cross-compiled by ops/build-llama-android.sh, which drops
# its .so files into ../jniLibs and refreshes the vendored headers in ./include.
cmake_minimum_required(VERSION 3.22)
project(agent_llama)

set(CMAKE_CXX_STANDARD 17)

set(LLAMA_LIB "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libllama.so")
if(NOT EXISTS "${LLAMA_LIB}")
    message(FATAL_ERROR "llama.cpp libraries not found for ${ANDROID_ABI}: run ops/build-llama-android.sh")
endif()

add_library(agent_llama SHARED agent_llama.cpp)
target_include_directories(agent_llama PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libllama.so)
add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml.so)

find_library(log-lib log)
target_link_libraries(agent_llama llama ggml ${log-lib})
target_compile_options(agent_llama PRIVATE -O3)
# 16 KB pages (Android 15+ devices).
target_link_options(agent_llama PRIVATE "-Wl,-z,max-page-size=16384")
