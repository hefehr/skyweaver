include_directories(${GTEST_INCLUDE_DIR})

link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_skyweaver_src
    src/AggregationBufferTester.cu
    src/BeamformerPipelineTester.cu
    src/BufferedDispenserTester.cu
    src/CoherentBeamformerTester.cu
    src/CoherentDedisperserTester.cu
    src/DelayManagerTester.cu
    src/ObservationHeaderTester.cpp
    src/IncoherentBeamformerTester.cu
    src/IncoherentDedisperserTester.cu
    src/IncoherentDedispersionPipelineTester.cu
    src/MultiFileReaderTester.cu
    src/MultiFileWriterTester.cu
    src/StatisticsCalculatorTester.cu
    src/TransposerTester.cu
    src/WeightsManagerTester.cu
    src/gtest_skyweaver.cu
)
cuda_add_executable(gtest_skyweaver ${gtest_skyweaver_src} )
target_link_libraries(gtest_skyweaver ${SKYWEAVER_LIBRARIES} ${GTEST_LIBRARIES} ${CUDA_cufft_LIBRARY} OpenMP::OpenMP_CXX)
add_test(NAME gtest_skyweaver 
         COMMAND gtest_skyweaver 
         WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")