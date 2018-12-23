#pragma once

#include "dnn.hpp"
#include <thread>  // std::hardware_concurrency 
#include <memory>  // unique_ptr
#include <tbb/task_scheduler_init.h>
#include <tbb/flow_graph.h>

inline void run_tbb(MNIST& D) {

  using namespace tbb;
  using namespace tbb::flow;

  tbb::task_scheduler_init init(std::thread::hardware_concurrency());
  tbb::flow::graph G;

  std::vector<std::unique_ptr<continue_node<continue_msg>>> forward_tasks;
  std::vector<std::unique_ptr<continue_node<continue_msg>>> backward_tasks;
  std::vector<std::unique_ptr<continue_node<continue_msg>>> update_tasks;
  std::vector<std::unique_ptr<continue_node<continue_msg>>> shuffle_tasks;


  std::vector<Eigen::MatrixXf> mats(D.num_storage, D.images);
  std::vector<Eigen::VectorXi> vecs(D.num_storage, D.labels);

  // Create task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;

  for(auto e=0; e<D.epoch; e++) {
    for(auto i=0; i<iter_num; i++) {
      auto& f_task = forward_tasks.emplace_back(
        std::make_unique<continue_node<continue_msg>>(G, 
          [&, iter = i, e=e%D.num_storage](const continue_msg&) {

            if(iter != 0) {
              D.beg_row += D.batch_size;
              if(D.beg_row >= D.images.rows()) {
                D.beg_row = 0;
              }
            }
            for(size_t i=0; i<D.acts.size(); i++) {
              if(i == 0){
                D.forward(i, mats[e].middleRows(D.beg_row, D.batch_size));
              }
              else {
                D.forward(i, D.Ys[i-1]);
              }
            }

            D.loss(vecs[e]);
          }
        ) 
      );

      if(i == 0 && e != 0) {
        auto sz = update_tasks.size();
        for(auto j=1; j<=D.acts.size() ;j++) {
          make_edge(*update_tasks[sz-j], *f_task);
        }         
      }

      if(i != 0) {
        auto sz = update_tasks.size();
        for(auto j=1; j<=D.acts.size() ;j++) {
          make_edge(*update_tasks[sz-j], *f_task);
        }
      }

      for(int j=D.acts.size()-1; j>=0; j--) {
        // backward propagation
        auto& b_task = backward_tasks.emplace_back(
          std::make_unique<continue_node<continue_msg>>(G, 
            [&, i=j, e=e%D.num_storage] (const continue_msg&) {
              if(i > 0) {
                D.backward(i, D.Ys[i-1].transpose());       
              }
              else {
                D.backward(i, mats[e].middleRows(D.beg_row, D.batch_size).transpose());
              }
            }
          )
        );
        // update weight 
        auto& u_task = update_tasks.emplace_back(
          std::make_unique<continue_node<continue_msg>>(G, [&, i=j] (const continue_msg&) {D.update(i);}));

        if(j == D.acts.size() - 1) {
          make_edge(*f_task, *b_task);
        }
        else {
          make_edge(*backward_tasks[backward_tasks.size()-2], *b_task);
          make_edge(*backward_tasks[backward_tasks.size()-2], *update_tasks[update_tasks.size()-2]);
        }
      } // End of backward propagation  
      make_edge(*backward_tasks.back(), *update_tasks.back());
    } // End of all iterations (task flow graph creation)


    if(e == 0) {
      // No need to shuffle in first epoch
      shuffle_tasks.emplace_back(std::make_unique<continue_node<continue_msg>>(G, [](const continue_msg&){}));
      make_edge(*shuffle_tasks.back(), *forward_tasks[forward_tasks.size()-iter_num]);
    }
    else {
      auto& t = shuffle_tasks.emplace_back(
        std::make_unique<continue_node<continue_msg>>(G, [&, e=e%D.num_storage](const continue_msg&) 
          {D.shuffle(mats[e], vecs[e], D.images.rows());})
      );
      make_edge(*t,*forward_tasks[forward_tasks.size()-iter_num]);

      if(e >= D.num_storage) {
        make_edge(*shuffle_tasks[e-D.num_storage] ,*t);
      }
    }
  } // End of all epoch

  for(size_t i=0; i<D.num_storage; i++) 
    shuffle_tasks[i]->try_put(continue_msg());
  G.wait_for_all();
}


