#pragma once

#include "dnn.hpp"
#include <taskflow/taskflow.hpp>  

inline void run_taskflow(MNIST& D) {
  tf::Taskflow tf {std::thread::hardware_concurrency()};

  std::vector<tf::Task> forward_tasks;
  std::vector<tf::Task> backward_tasks;
  std::vector<tf::Task> update_tasks;
  std::vector<tf::Task> shuffle_tasks;

  // Number of parallel shuffle
  const auto num_par_shf = std::min(D.num_storage, D.epoch);

  std::vector<Eigen::MatrixXf> mats(num_par_shf, D.images);
  std::vector<Eigen::VectorXi> vecs(num_par_shf, D.labels);

  // Create task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;

  for(auto e=0; e<D.epoch; e++) {
    for(auto i=0; i<iter_num; i++) {
      auto& f_task = forward_tasks.emplace_back(
        tf.silent_emplace(
          [&, iter = i, e=e%num_par_shf]() {
            forward_task(D, i, e, mats, vecs);
          }
        ) 
      ).name("e" + std::to_string(e) + "_F"+std::to_string(i));

      if(i != 0 || (i == 0 && e != 0)) {
        auto sz = update_tasks.size();
        for(auto j=1; j<=D.acts.size() ;j++) {
          update_tasks[sz-j].precede(f_task);
        }         
      }

      for(int j=D.acts.size()-1; j>=0; j--) {
        // backward propagation
        auto& b_task = backward_tasks.emplace_back(
          tf.silent_emplace(
            [&, i=j, e=e%num_par_shf] () {
              backward_task(D, i, e, mats);
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
          backward_tasks[backward_tasks.size()-2].precede(b_task);
        }
        b_task.precede(u_task);
      } // End of backward propagation 
    } // End of all iterations (task flow graph creation)


    if(e == 0) {
      // No need to shuffle in first epoch
      shuffle_tasks.emplace_back(tf.silent_emplace([](){}))
                   .precede(forward_tasks[forward_tasks.size()-iter_num])           
                   .name("e" + std::to_string(e) + "_S" + std::to_string(e%num_par_shf));
    }
    else {
      auto& t = shuffle_tasks.emplace_back(
        tf.silent_emplace([&, e=e%num_par_shf]() {D.shuffle(mats[e], vecs[e], D.images.rows());})
      ).precede(forward_tasks[forward_tasks.size()-iter_num])           
       .name("e" + std::to_string(e) + "_S" + std::to_string(e%num_par_shf));

      // This shuffle task starts after belows finish
      //   1. previous shuffle on the same storage
      //   2. the last backward task of previous epoch on the same storage 
      if(e >= num_par_shf) {
        auto prev_e = e - num_par_shf;
        shuffle_tasks[prev_e].precede(t);

        int task_id = (prev_e+1)*iter_num*D.acts.size() - 1;
        backward_tasks[task_id].precede(t);
      }
    }
  } // End of all epoch

  tf.wait_for_all();
}

