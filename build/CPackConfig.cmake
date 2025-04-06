# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "/Users/mandukhaialimaa/UWMadison/744/research/quake;/Users/mandukhaialimaa/UWMadison/744/research/quake/build")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENTS_ALL "Devel;Unspecified")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Cameron Desrochers <cameron@moodycamel.com>")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/opt/homebrew/Caskroom/miniconda/base/envs/quake-env/share/cmake-3.23/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "quake_c built using CMake")
set(CPACK_GENERATOR "RPM;DEB")
set(CPACK_INSTALL_CMAKE_PROJECTS "/Users/mandukhaialimaa/UWMadison/744/research/quake/build;quake_c;ALL;/")
set(CPACK_INSTALL_PREFIX "/usr/local")
set(CPACK_MODULE_PATH "")
set(CPACK_NSIS_DISPLAY_NAME "concurrentqueue 1.0.0")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "concurrentqueue 1.0.0")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk")
set(CPACK_OUTPUT_CONFIG_FILE "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/opt/homebrew/Caskroom/miniconda/base/envs/quake-env/share/cmake-3.23/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "An industrial-strength lock-free queue for C++.")
set(CPACK_PACKAGE_FILE_NAME "concurrentqueue-1.0.0-Darwin")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "concurrentqueue 1.0.0")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "concurrentqueue 1.0.0")
set(CPACK_PACKAGE_NAME "concurrentqueue")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Cameron Desrochers <cameron@moodycamel.com>")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_RESOURCE_FILE_LICENSE "/opt/homebrew/Caskroom/miniconda/base/envs/quake-env/share/cmake-3.23/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "/opt/homebrew/Caskroom/miniconda/base/envs/quake-env/share/cmake-3.23/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "/opt/homebrew/Caskroom/miniconda/base/envs/quake-env/share/cmake-3.23/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TBZ2;TGZ;TXZ;TZ")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CPackSourceConfig.cmake")
set(CPACK_SOURCE_RPM "OFF")
set(CPACK_SOURCE_TBZ2 "ON")
set(CPACK_SOURCE_TGZ "ON")
set(CPACK_SOURCE_TXZ "ON")
set(CPACK_SOURCE_TZ "ON")
set(CPACK_SOURCE_ZIP "OFF")
set(CPACK_SYSTEM_NAME "Darwin")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Darwin")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
