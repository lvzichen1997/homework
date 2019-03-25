#声明要求的cmake最低版本
cmake_minimum_required(VERSION 2.6)
#声明一个cmake工程
project(compkeypointextract)
#添加c++11标准支持
add_definitions(-std=c++11)
#寻找opencv库
find_package(OpenCV REQUIRED)
#添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
#添加可执行程序
add_executable(compkeypointextract main.cpp)
#将可执行文件链接到库上
target_link_libraries(compkeypointextract ${OpenCV_LIBS})
install(TARGETS compkeypointextract RUNTIME DESTINATION bin)
