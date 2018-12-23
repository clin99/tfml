#pragma once

#include "dnn.hpp"
#include <taskflow/taskflow.hpp>  

inline void run_taskflow(MNIST& D) {
  tf::Taskflow tf {4};

  std::vector<tf::Task> forward_tasks;
  std::vector<tf::Task> backward_tasks;
  std::vector<tf::Task> update_tasks;
  std::vector<tf::Task> shuffle_tasks;

  std::vector<Eigen::MatrixXf> mats(D.num_storage, D.images);
  std::vector<Eigen::VectorXi> vecs(D.num_storage, D.labels);

  // Create task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;

  for(auto e=0; e<D.epoch; e++) {
    for(auto i=0; i<iter_num; i++) {
      auto& f_task = forward_tasks.emplace_back(
        tf.silent_emplace(
          [&, iter = i, e=e%D.num_storage]() {

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
      ).name("e" + std::to_string(e) + "_F"+std::to_string(i));

      if(i == 0 && e != 0) {
        auto sz = update_tasks.size();
        for(auto j=1; j<=D.acts.size() ;j++) {
          update_tasks[sz-j].precede(f_task);
        }         
      }

      if(i != 0) {
        auto sz = update_tasks.size();
        for(auto j=1; j<=D.acts.size() ;j++) {
          update_tasks[sz-j].precede(f_task);
        }
      }

      for(int j=D.acts.size()-1; j>=0; j--) {
        // backward propagation
        auto& b_task = backward_tasks.emplace_back(
          tf.silent_emplace(
            [&, i=j, e=e%D.num_storage] () {
              if(i > 0) {
                D.backward(i, D.Ys[i-1].transpose());       
              }
              else {
                D.backward(i, mats[e].middleRows(D.beg_row, D.batch_size).transpose());
              }
            }
          )
        ).name("e" + std::to_string(e) + "_L" + std::to_string(i) +"_B" + std::to_string(j));
        // update weight 
        auto& u_task = update_tasks.emplace_back(
          tf.silent_emplace([&, i=j] () {D.update(i);})
        ).name("e" + std::to_string(e) + "_L" + std::to_string(i) +"_U" + std::to_string(j));

        if(j == D.acts.size() - 1) {
          f_task.precede(b_task);
        }
        else {
          backward_tasks[backward_tasks.size()-2].precede(b_task).precede(update_tasks[update_tasks.size()-2]);
        }
      } // End of backward propagation 
      backward_tasks.back().precede(update_tasks.back()); 
    } // End of all iterations (task flow graph creation)


    if(e == 0) {
      // No need to shuffle in first epoch
      shuffle_tasks.emplace_back(tf.silent_emplace([](){}))
                   .precede(forward_tasks[forward_tasks.size()-iter_num])           
                   .name("e" + std::to_string(e) + "_S" + std::to_string(e%D.num_storage));
    }
    else {
      auto& t = shuffle_tasks.emplace_back(
        tf.silent_emplace([&, e=e%D.num_storage]() {D.shuffle(mats[e], vecs[e], D.images.rows());})
      ).precede(forward_tasks[forward_tasks.size()-iter_num])           
       .name("e" + std::to_string(e) + "_S" + std::to_string(e%D.num_storage));

      if(e >= D.num_storage) {
        shuffle_tasks[e-D.num_storage].precede(t);
      }
    }
  } // End of all epoch

  tf.wait_for_all();
}

