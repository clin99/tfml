#include "seq.hpp"
#include "tbb.hpp"
#include "tf.hpp"
#include "openmp.hpp"

#define BENCHMARK(LIB)                                                 \
  {                                                                    \
    auto dnn {build_dnn()};                                            \
    std::cout << "Benchmark " #LIB << '\n';                            \
    auto t1 = std::chrono::high_resolution_clock::now();               \
    run_##LIB(dnn);                                                    \
    auto t2 = std::chrono::high_resolution_clock::now();               \
    std::cout << "Benchmark runtime: " << time_diff(t1, t2) << " s\n"; \
    dnn.validate(); \
  }



int main(int argc, char *argv[]){

  ////run_omp4();
  //BENCHMARK(omp6);
  //BENCHMARK(tbb); 
  ////BENCHMARK(sequential);
  ////BENCHMARK(taskflow);
  //return 0;
  
  int sel = 0;
  if(argc > 1) {
    if(::strcmp(argv[1], "taskflow") == 0) {
      sel = 1;
    }
    else if(::strcmp(argv[1], "tbb") == 0) {
      sel = 2;
    }
    else if(::strcmp(argv[1], "omp") == 0) {
      sel = 3;
    }
  }

  switch(sel) {
    case 1:  BENCHMARK(taskflow);   break; 
    case 2:  BENCHMARK(tbb);        break;
    case 3:  BENCHMARK(omp);        break;
    default: BENCHMARK(sequential); break;
  };

  return EXIT_SUCCESS;
}

