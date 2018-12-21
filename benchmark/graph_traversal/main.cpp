#include "levelgraph.hpp"

int main() {

  double omp_time {0.0};
  double tbb_time {0.0};
  double tf_time  {0.0};
  int rounds {5};
 
  std::cout << std::setw(12) << "|V|+|E|"
            << std::setw(12) << "OpenMP"
            << std::setw(12) << "TBB"
            << std::setw(12) << "Taskflow"
            << std::endl;

  for(int i=1; i<=401; i += 4) {

    LevelGraph graph(i, i);
    
    for(int j=0; j<rounds; ++j) {
      omp_time += measure_time_omp(graph).count();
      graph.clear_graph();

      tbb_time += measure_time_tbb(graph).count();
      graph.clear_graph();

      tf_time  += measure_time_taskflow(graph).count();
      graph.clear_graph();
    }

    std::cout << std::setw(12) << graph.graph_size() 
              << std::setw(12) << omp_time / rounds / 1000.0
              << std::setw(12) << tbb_time / rounds / 1000.0
              << std::setw(12) << tf_time  / rounds / 1000.0
              << std::endl;
  }
}


