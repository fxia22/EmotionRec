cmake_minimum_required(VERSION 2.8)
project( FACETRC )  
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBS_DIR})


find_package( OpenCV REQUIRED ) 
ADD_SUBDIRECTORY(src bin)
