set -x
#g++-8 -std=c++17 ./unittest/threadpool.cpp -O2 -lpthread -I ./doctest -I .
#g++-8 -std=c++17 ./example/threadpool.cpp -O2 -lpthread -I ./doctest -I .
#g++-8 -std=c++17 ./example/taskflow.cpp -O2 -lpthread -I ./doctest -I .

#g++-8 -std=c++17 ./example/dnn.cpp -O2 -lpthread -I ./doctest -I . -lstdc++fs

g++-8 -ggdb3 -O2 -I ./eigen -std=c++17 ./example/dnn.cpp -lpthread -I ./doctest -I . -lstdc++fs
