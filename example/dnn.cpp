#include "dnn.hpp"

#include "tbb.hpp"
#include "tf.hpp"
#include "seq.hpp"

#define BENCHMARK(LIB)                                                 \
  {                                                                    \
    auto dnn {build_dnn()};                                            \
    printf("Benchmark " #LIB "\n");                                    \
    auto t1 = std::chrono::high_resolution_clock::now();               \
    run_##LIB(dnn);                                                    \
    auto t2 = std::chrono::high_resolution_clock::now();               \
    std::cout << "Benchmark runtime: " << time_diff(t1, t2) << " s\n"; \
    dnn.validate(); \
  }



int main(int argc, char *argv[]){
  
  int sel = 0;
  if(argc > 1) {
    if(::strcmp(argv[1], "taskflow") == 0) {
      sel = 1;
    }
    else if(::strcmp(argv[1], "tbb") == 0) {
      sel = 2;
    }
  }

  switch(sel) {
    case 1:  BENCHMARK(taskflow);   break; 
    case 2:  BENCHMARK(tbb);        break;
    default: BENCHMARK(sequential); break;
  };

  return 0;
}

