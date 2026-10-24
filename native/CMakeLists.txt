cmake_minimum_required(VERSION 3.24)
project(KaptiaNative VERSION 0.1.0 LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "Kaptia Native bootstrap currently targets Windows 10/11 x64.")
endif()

add_executable(kaptia-native
  src/main.cpp
  src/diagnostics.cpp
  src/downloader.cpp
)
target_compile_features(kaptia-native PRIVATE cxx_std_20)
target_compile_definitions(kaptia-native PRIVATE UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)
target_link_libraries(kaptia-native PRIVATE winhttp)

if(MSVC)
  target_compile_options(kaptia-native PRIVATE /W4 /permissive- /EHsc)
endif()
