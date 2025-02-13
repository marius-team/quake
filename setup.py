import os
import platform
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            raise RuntimeError("Unsupported on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, "-DPYTHON_EXECUTABLE=" + sys.executable]

        # cfg = "Debug" if self.debug else "Release"
        cfg = "Debug"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            raise RuntimeError("Unsupported on Windows")
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

            num_threads = os.cpu_count()
            build_args += ["--", "-j{}".format(num_threads)]

        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        cmake_args += ["-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE"]

        try:
            import torch
        except ImportError:
            raise ImportError("Pytorch not found. Please install pytorch first.")

        if sys.platform == "darwin":
            cmake_args.append("-DCMAKE_INSTALL_RPATH=@loader_path")
        else:  # values: linux*, aix, freebsd, ... just as well win32 & cygwin
            cmake_args.append("-DCMAKE_INSTALL_RPATH=$ORIGIN")

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get("CXXFLAGS", ""), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print("Building Quake:")
        print(cmake_args)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", ".", "--target", "bindings"] + build_args, cwd=self.build_temp)

setup(
    name="quake",
    version="0.0.1",
    ext_modules=[CMakeExtension("quake._bindings")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)