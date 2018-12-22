#include "dnn.hpp"

int main(int argc, char *argv[]){
  
  int sel = 0;
  if(argc > 1) {
    if(::strcmp(argv[1], "taskflow") == 0) {
      sel = 1;
    }
    else if(::strcmp(argv[1], "tbb") == 0) {
      assert(false); 
      sel = 2;
    }
  }

  switch(sel) {
    case 1:  measure_taskflow(); break;
    default: measure_sequential(); break;
  };

  return 0;
}


