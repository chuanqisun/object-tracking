
# Clone and build uSockets & uWebSockets
git clone --recursive https://github.com/uNetworking/uSockets.git
cd uSockets
make
sudo cp src/libusockets.h /usr/include/
sudo cp uSockets.a /usr/lib64/libusockets.a
cd ..

git clone --recursive https://github.com/uNetworking/uWebSockets.git
sudo cp -r uWebSockets/src/* /usr/include/