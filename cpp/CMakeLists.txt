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
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  find_package(nlohmann_json REQUIRED)
  add_executable(mocap_camd node/main.cpp node/camera.cpp node/h264_encoder.cpp node/frame_prep.cpp
                          node/http_server.cpp)
  target_link_libraries(mocap_camd PRIVATE ${OpenCV_LIBS}
                        nlohmann_json::nlohmann_json Threads::Threads)
  install(TARGETS mocap_camd DESTINATION bin)
endif()

if(MOCAP_BUILD_LAPTOP)
  pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavcodec libavutil)
  add_executable(mocap_camstat tools/camstat.cpp)
  target_link_libraries(mocap_camstat PRIVATE PkgConfig::LIBAV Threads::Threads)
  # The receiving path: nodes -> decoded frames (library for the C++ tract, and mocap_rx).
  add_library(mocap_rx_lib STATIC rx/node_receiver.cpp rx/shm_ring.cpp rx/recorder.cpp)
  target_link_libraries(mocap_rx_lib PUBLIC PkgConfig::LIBAV Threads::Threads rt)
  add_executable(mocap_rx rx/main.cpp)
  target_link_libraries(mocap_rx PRIVATE mocap_rx_lib)
  install(TARGETS mocap_camstat mocap_rx DESTINATION bin)
endif()
