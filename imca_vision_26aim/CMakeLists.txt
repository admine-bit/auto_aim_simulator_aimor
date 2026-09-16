cmake_minimum_required(VERSION 3.16.3)

project(imca_vision)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

# Keep the default build independent from an activated Conda environment.
# CMake searches every "bin" entry in PATH as a package prefix, so an active
# Conda shell can otherwise mix Conda fmt/yaml-cpp/nlohmann_json with the
# system spdlog library. That combination is ABI/API incompatible on this host.
option(IMCA_ALLOW_CONDA_PACKAGES "Allow dependencies to be resolved from Conda environments" OFF)

if(NOT IMCA_ALLOW_CONDA_PACKAGES)
    set(_imca_conda_prefixes "$ENV{CONDA_PREFIX}" "$ENV{_CONDA_ROOT}")

    if(DEFINED ENV{PATH})
        string(REPLACE ":" ";" _imca_path_entries "$ENV{PATH}")
        foreach(_imca_path_entry IN LISTS _imca_path_entries)
            if(NOT _imca_path_entry)
                continue()
            endif()

            get_filename_component(_imca_path_name "${_imca_path_entry}" NAME)
            if(_imca_path_name STREQUAL "bin" OR _imca_path_name STREQUAL "condabin")
                get_filename_component(_imca_path_prefix "${_imca_path_entry}" DIRECTORY)
                if(EXISTS "${_imca_path_prefix}/conda-meta" OR
                   EXISTS "${_imca_path_prefix}/condabin/conda")
                    list(APPEND _imca_conda_prefixes "${_imca_path_prefix}")
                endif()
            endif()
        endforeach()
    endif()

    list(FILTER _imca_conda_prefixes EXCLUDE REGEX "^$")
    list(REMOVE_DUPLICATES _imca_conda_prefixes)
    if(_imca_conda_prefixes)
        list(APPEND CMAKE_IGNORE_PREFIX_PATH ${_imca_conda_prefixes})
        list(REMOVE_DUPLICATES CMAKE_IGNORE_PREFIX_PATH)
        message(STATUS "Ignoring Conda package prefixes: ${_imca_conda_prefixes}")
    endif()

    # A previous configure may have persisted package *_DIR entries below a
    # Conda prefix. Drop only those stale entries; preserve all other explicit
    # package paths supplied by the user. CMake's generated target metadata can
    # still contain the old imported library paths, so require one fresh
    # regeneration instead of allowing a misleading incremental build.
    set(_imca_discarded_conda_cache FALSE)
    foreach(_imca_package_dir IN ITEMS
            OpenCV_DIR fmt_DIR Eigen3_DIR spdlog_DIR yaml-cpp_DIR
            nlohmann_json_DIR Ceres_DIR OpenVINO_DIR)
        if(NOT DEFINED ${_imca_package_dir})
            continue()
        endif()

        set(_imca_probe "${${_imca_package_dir}}")
        set(_imca_cached_from_conda FALSE)
        while(IS_ABSOLUTE "${_imca_probe}")
            if(EXISTS "${_imca_probe}/conda-meta")
                set(_imca_cached_from_conda TRUE)
                break()
            endif()

            get_filename_component(_imca_parent "${_imca_probe}" DIRECTORY)
            if("${_imca_parent}" STREQUAL "${_imca_probe}")
                break()
            endif()
            set(_imca_probe "${_imca_parent}")
        endwhile()

        if(_imca_cached_from_conda)
            message(STATUS "Discarding cached Conda package path: ${_imca_package_dir}=${${_imca_package_dir}}")
            set(_imca_discarded_conda_cache TRUE)
            unset(${_imca_package_dir} CACHE)
            unset(${_imca_package_dir})
        endif()
    endforeach()

    if(_imca_discarded_conda_cache)
        message(FATAL_ERROR
            "This build directory was previously configured with Conda packages. "
            "Regenerate it once with: cmake --fresh -B \"${CMAKE_BINARY_DIR}\". "
            "After that, the normal cmake -B build and make -C build commands are sufficient.")
    endif()
endif()

include(cmake/inference_backend.cmake)

if(IMCA_INFERENCE_BACKEND STREQUAL "TENSORRT")
    # TensorRT version uses an OpenCV build with CUDA modules.
    set(OpenCV_DIR "/opt/opencv-cuda/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Ceres REQUIRED)

message(STATUS "Dependency package roots:")
message(STATUS "  OpenCV: ${OpenCV_DIR}")
message(STATUS "  fmt: ${fmt_DIR}")
message(STATUS "  spdlog: ${spdlog_DIR}")
message(STATUS "  yaml-cpp: ${yaml-cpp_DIR}")
message(STATUS "  nlohmann_json: ${nlohmann_json_DIR}")
message(STATUS "  Ceres: ${Ceres_DIR}")

# 两种构建互斥，Jetson TensorRT 版本不查找 OpenVINO。
if(IMCA_INFERENCE_BACKEND STREQUAL "OPENVINO")
    # Prefer the host SDK. Conda package paths are accepted only when the build
    # is explicitly configured with IMCA_ALLOW_CONDA_PACKAGES=ON.
    if(NOT OpenVINO_DIR)
        set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
    endif()
    find_package(OpenVINO REQUIRED)
    add_compile_definitions(IMCA_BACKEND_OPENVINO)
elseif(IMCA_INFERENCE_BACKEND STREQUAL "TENSORRT")
    find_package(CUDAToolkit REQUIRED)
    set(TensorRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
    find_library(NVINFER nvinfer)
    if(NOT NVINFER)
        message(FATAL_ERROR "TensorRT library nvinfer was not found")
    endif()
    add_compile_definitions(IMCA_BACKEND_TENSORRT)
else()
    message(FATAL_ERROR "IMCA_INFERENCE_BACKEND must be OPENVINO or TENSORRT")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

if(IMCA_INFERENCE_BACKEND STREQUAL "TENSORRT")
    include_directories(${TensorRT_INCLUDE_DIR})
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(rm_vision_core-main)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

####################src###################
add_executable(standard_mpc src/standard_mpc.cpp)
add_executable(auto_aim_debug_mpc src/auto_aim_debug_mpc.cpp)
add_executable(auto_buff_debug_yolo src/auto_buff_debug_yolo.cpp)
add_executable(auto_buff_debug_mpc src/auto_buff_debug_mpc.cpp)
add_executable(uav src/uav.cpp)
add_executable(uav_debug src/uav_debug.cpp)

target_link_libraries(standard_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(auto_aim_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(auto_buff_debug_yolo ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(auto_buff_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(uav ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(uav_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

##################calibration################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_fast_camera calibration/calibrate_fast_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_fast_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools Ceres::ceres)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)

##################tests##################
add_executable(auto_aim_test tests/auto_aim_test.cpp)
# Buff 离线验收统一使用 auto_buff_test；旧 auto_buff_yolo_test 实际运行的是装甲板链路，故不再注册。
add_executable(auto_buff_test tests/auto_buff_test.cpp)
add_executable(camera_detect_test tests/camera_detect_test.cpp)
add_executable(camera_test tests/camera_test.cpp)
add_executable(camera_thread_test tests/camera_thread_test.cpp)
add_executable(cboard_test tests/cboard_test.cpp)
add_executable(fire_test tests/fire_test.cpp)
add_executable(detector_video_test tests/detector_video_test.cpp)
add_executable(gimbal_response_test tests/gimbal_response_test.cpp)
add_executable(multi_usbcamera_test tests/multi_usbcamera_test.cpp)
add_executable(usbcamera_detect_test tests/usbcamera_detect_test.cpp)
add_executable(usbcamera_test tests/usbcamera_test.cpp)
add_executable(handeye_test tests/handeye_test.cpp)
add_executable(dm_test tests/dm_test.cpp)
add_executable(minimum_vision_system tests/minimum_vision_system.cpp)

target_link_libraries(auto_aim_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_sources(auto_buff_test PRIVATE $<TARGET_OBJECTS:auto_aim>)
# 与 auto_buff_debug_mpc 保持同一组依赖：rm-core 调试窗口来自 auto_buff 传递的
# VisCore 目标，Ceres 由 buff_solver 需要。
target_link_libraries(auto_buff_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(camera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim io)
target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(camera_thread_test ${OpenCV_LIBS} fmt::fmt auto_aim tools io)
target_link_libraries(cboard_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(fire_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(detector_video_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim)
target_link_libraries(gimbal_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(multi_usbcamera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(usbcamera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(usbcamera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(handeye_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(dm_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(minimum_vision_system ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)

add_executable(gimbal_test tests/gimbal_test.cpp)
target_link_libraries(gimbal_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test tests/planner_test.cpp)
target_link_libraries(planner_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test_offline tests/planner_test_offline.cpp)
target_link_libraries(planner_test_offline ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

# 检查 ROS 环境
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)
# 只有在找到 ROS 环境时才会添加 sentry 和 publish_test
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
    # 编译 sentry 和 publish_test
    add_executable(sentry src/sentry.cpp)
    add_executable(sentry_debug src/sentry_debug.cpp)
    add_executable(sentry_multithread src/sentry_multithread.cpp)
    add_executable(publish_test tests/publish_test.cpp)
    add_executable(subscribe_test tests/subscribe_test.cpp)
    add_executable(topic_loop_test tests/topic_loop_test.cpp)

    target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_multithread ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim)
    target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
    target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)
else()
    message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
endif()
