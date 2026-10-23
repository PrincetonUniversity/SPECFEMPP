# Standalone plugin project built against the *installed* SDK only, exactly as
# an external plugin author would. Driven by run_consumer.cmake.

cmake_minimum_required(VERSION 3.17)
project(specfempp_plugin_sdk_consumer LANGUAGES CXX)

find_package(SpecfemppPluginSDK 1 REQUIRED)

specfempp_add_injection_plugin(consumer_plugin SOURCES consumer_plugin.cpp)
