from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout

class ParallelConan(ConanFile):
    name = "LearnProject"
    version = "1.0.0"
    license = "Apache License v2.0"
    author = "Michael Pohl"
    description = "An exploratory project for a C++ application which has a library for neural networks"
    topics = ()
    settings = "os", "compiler", "build_type", "arch"
    requires = [
        "catch2/3.1.0",
        "clara/1.1.5",
        "range-v3/0.10.0",
    ]
    options = {"coverage": [True, False], "formatting": [True, False]}
    default_options = {"coverage": False, "formatting": False}

    def configure(self):
        print("do nothing in configure")

    def requirements(self):
        print("do nothing in requirements")

    def layout(self):
        cmake_layout(self, src_folder=".")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        coverage = getattr(self.options, "coverage", False)
        formatting = getattr(self.options, "formatting", False)

        if coverage:
            self.output.info("Building with coverage flags...")
            self._build_with_coverage()
        elif formatting:
            self.output.info("Building with formatting flags...")
            self._build_with_formatting()
        else:
            self.output.info("Building without coverage flags...")
            self._build_without_coverage()

    def _build_with_coverage(self):
        cmake = CMake(self)
        vars = {
            "ENABLE_COVERAGE": "true",
        }
        cmake.configure(vars)
        cmake.build()

    def _build_with_formatting(self):
        cmake = CMake(self)
        vars = {
            "ENABLE_FORMATTING": "true",
        }
        cmake.configure(vars)
        cmake.build()

    def _build_without_coverage(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
