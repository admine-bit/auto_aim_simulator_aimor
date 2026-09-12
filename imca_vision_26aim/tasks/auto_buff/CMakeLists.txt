cmake_minimum_required(VERSION 3.16)

find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT
    buff_type.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_aimer.cpp
    buff_target.cpp
    buff_debug_draw.cpp
    rm_core_adapter.cpp
    yolo11_buff.cpp
)

set_target_properties(auto_buff PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

target_link_libraries(auto_buff auto_aim VisCore_rune_detector ${CERES_LIBRARIES})
