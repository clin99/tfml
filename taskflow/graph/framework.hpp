#pragma once

#include "flow_builder.hpp"

namespace tf {

// Class: Framework
class Framework : public FlowBuilder {
  
  template <template<typename...> typename E> 
  friend class BasicTaskflow;
  friend class TopologyView;  

  public:

    inline Framework();

  protected:

    Graph _graph;

};

// Constructor
inline Framework::Framework() : FlowBuilder{_graph} {
}


class TopologyView {
  template <template<typename...> typename E>
  friend class BasicTaskflow;

  public:
    TopologyView(Framework&, size_t=1);

  private:
    Framework& _framework;
    std::vector<Node*> _sources;
    std::forward_list<std::shared_future<void>> _futures;
    size_t _repeat {1};
    Node _target;
};

inline TopologyView::TopologyView(Framework& framework, size_t repeat): 
  _framework(framework), 
  _repeat(repeat){
  for(auto&n : framework._graph) {
    if(n.num_dependents() == 0) {
      _sources.emplace_back(&n);
    }
    if(n.num_successors() == 0) {
      n.precede(_target);
    }
  }
}

};  // end of namespace tf. ---------------------------------------------------



