
To install the repository, find your desired working directory and run the following shell command:
```shell
git clone https://github.com/Malav-P/CNN.git
```

Next, `cd` into the `CNN` directory and make a directory called `build`. Navigate into this directory and run 
the cmake commands. The outline is shown below.

```sh
cd CNN
mkdir build
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=<desired installation path>
make
make install
```

When using this package in your own project, include the following in your `CMakeLists.txt` file:
```
find_package(CNN REQUIRED)
target_link_libraries(<your program here> PRIVATE cnn)
```
AND, when building your project, be sure to specify `CMAKE_PREFIX_PATH` variable to correspond to the
`<desired installation path>` that was chosen above. For example, a project that utilizes this library 
and installed the library at `/usr/local/` would call cmake like so:
```shell
cmake .. -DCMAKE_PREFIX_PATH=/usr/local
```