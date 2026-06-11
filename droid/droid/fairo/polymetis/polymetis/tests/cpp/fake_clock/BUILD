# This BUILD file uses Bazel. See http://bazel.io/
# If you don't want to use Bazel, compiling the lib is easy.
#   $ g++ -std=c++11 fake_clock.cc

package(
    default_visibility = ["//visibility:private"])

cc_library(
    name = "fake_clock_lib",
    srcs = ["fake_clock.cc"],
    hdrs = ["fake_clock.hh"],
    visibility = ["//visibility:public"])

cc_test(
    name = "fake_clock_test",
    srcs = ["fake_clock_test.cc"],
    deps = [
        ":fake_clock_lib",
        "@gtest//gtest:gtest_main",
    ])
