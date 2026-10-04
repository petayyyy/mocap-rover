cmake_minimum_required(VERSION 3.16)
project(mocap_rover_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# The camera node talks V4L2 to the CM4's unicam and encoder: build it on the node.
option(MOCAP_BUILD_NODE "camera node (mocap_camd), needs OpenCV imgproc" OFF)
# Laptop-side tools: receiver/measurement, needs libavcodec.
option(MOCAP_BUILD_LAPTOP "laptop tools (mocap_camstat), needs libavcodec" ON)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

if(MOCAP_BUILD_NODE)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc)
  find_package(nlohmann_json REQUIRED)
  add_executable(mocap_camd node/main.cpp node/camera.cpp node/h264_encoder.cpp)
  target_link_libraries(mocap_camd PRIVATE ${OpenCV_LIBS}
                        nlohmann_json::nlohmann_json Threads::Threads)
  install(TARGETS mocap_camd DESTINATION bin)
endif()

if(MOCAP_BUILD_LAPTOP)
  pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavcodec libavutil)
  add_executable(mocap_camstat tools/camstat.cpp)
  target_link_libraries(mocap_camstat PRIVATE PkgConfig::LIBAV Threads::Threads)
  install(TARGETS mocap_camstat DESTINATION bin)
endif()
