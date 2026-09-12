cmake_minimum_required(VERSION 3.16)

add_subdirectory(planner/tinympc)

set(AUTO_AIM_SOURCES
    armor.cpp
    classifier.cpp
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    inference/infer_engine.cpp
    multithread/commandgener.cpp
    planner/planner.cpp
)

# 仅将当前选择的后端源文件加入目标。
if(IMCA_INFERENCE_BACKEND STREQUAL "OPENVINO")
    list(APPEND AUTO_AIM_SOURCES
        inference/openvino_engine.cpp
        multithread/mt_detector.cpp
    )
elseif(IMCA_INFERENCE_BACKEND STREQUAL "TENSORRT")
    list(APPEND AUTO_AIM_SOURCES
        inference/trt_engine.cpp
        multithread/mt_detector_tensorrt.cpp
    )
endif()

add_library(auto_aim OBJECT ${AUTO_AIM_SOURCES})

target_link_libraries(auto_aim io tinympcstatic)

# 仅链接当前后端需要的运行时库。
if(IMCA_INFERENCE_BACKEND STREQUAL "OPENVINO")
    target_link_libraries(auto_aim openvino::runtime)
elseif(IMCA_INFERENCE_BACKEND STREQUAL "TENSORRT")
    target_link_libraries(auto_aim ${NVINFER} CUDA::cudart)
endif()
