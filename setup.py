import os
import platform
import shutil
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

        # self.generate_stubs()

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, "-DPYTHON_EXECUTABLE=" + sys.executable]

        cfg = "Debug" if self.debug else "Release"
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

        # if the gpu version of torch is installed, add the flag to the cmake args to enable the GPU build
        if torch.cuda.is_available():
            cmake_args += ["-DQUAKE_ENABLE_GPU=ON"]
        else:
            cmake_args += ["-DQUAKE_ENABLE_GPU=OFF", "-DTorch_NO_CUDA=ON", "-DTorch_USE_CUDA=OFF", "-DUSE_CUDA=OFF"]
        # check if numa is available
        try:
            subprocess.check_output(["numactl", "--show"])
            cmake_args += ["-DQUAKE_USE_NUMA=ON"]
        except OSError:
            cmake_args += ["-DQUAKE_USE_NUMA=OFF"]

        # check if avx512 is supported by parsing lscpu output
        try:
            lscpu_output = subprocess.check_output(["lscpu"], universal_newlines=True)
            if "avx512" in lscpu_output.lower():
                cmake_args += ["-DQUAKE_USE_AVX512=ON"]
            else:
                cmake_args += ["-DQUAKE_USE_AVX512=OFF"]
        except OSError:
            cmake_args += ["-DQUAKE_USE_AVX512=OFF"]

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

    def generate_stubs(self):
        # Get the full path to the built extension.
        ext = self.extensions[0]
        ext_fullpath = self.get_ext_fullpath(ext.name)
        # ext_output_dir is the 'quake' directory.
        ext_output_dir = os.path.abspath(os.path.dirname(ext_fullpath))
        # The parent directory that contains 'quake'.
        package_parent_dir = os.path.dirname(ext_output_dir)

        print(f"Generating stubs for {ext.name} in {ext_output_dir}")

        # Add the parent directory to PYTHONPATH so that "import quake" works.
        env = os.environ.copy()
        env["PYTHONPATH"] = package_parent_dir + os.pathsep + env.get("PYTHONPATH", "")

        # Create a temporary directory for stub generation.
        stub_output_dir = os.path.join(self.build_temp, "stubs")
        os.makedirs(stub_output_dir, exist_ok=True)

        # Run stubgen on your module.
        cmd = ["pybind11-stubgen", "-o", stub_output_dir, "quake._bindings"]
        subprocess.check_call(cmd, env=env)

        # The generated stub should be at <stub_output_dir>/quake/_bindings.pyi.
        generated_stub = os.path.join(stub_output_dir, "quake", "_bindings.pyi")
        if not os.path.exists(generated_stub):
            raise RuntimeError("Stub generation failed; expected file not found.")

        # Copy the generated stub into your package directory so it gets installed.
        dest = os.path.join(ext_output_dir, "_bindings.pyi")
        shutil.copyfile(generated_stub, dest)
        print(f"Stub file copied to {dest}")


setup(
    name="quake",
    version="0.0.1",
    ext_modules=[CMakeExtension("quake._bindings")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
