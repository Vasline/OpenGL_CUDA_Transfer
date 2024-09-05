pushd %~dp0..
set GLFW_HOME=%cd%

set BUILD_DIR=%GLFW_HOME%\build_vs2015_win64
set OUTPUT_DIR=%GLFW_HOME%\out\win64

md %BUILD_DIR%
pushd %BUILD_DIR%

cmake .. ^
    -G "Visual Studio 14 2015" -A x64 ^
    -DUSE_MSVC_RUNTIME_LIBRARY_DLL=OFF

cmake --build . --config Release

md %OUTPUT_DIR%
copy /y %BUILD_DIR%\src\Release\glfw3.lib %OUTPUT_DIR%\glfw3.lib

popd
popd