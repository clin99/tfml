#pragma once

#include "dnn.hpp"
#include <omp.h>

void omp_forward_task(MNIST& D, size_t iter, size_t e, 
  std::vector<Eigen::MatrixXf>& mats, 
  std::vector<Eigen::VectorXi>& vecs) {
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
  //printf("\x1B[31mF %d %d\x1B[0m\n", iter, e);
}

void omp_backward_task(MNIST& D, size_t i, size_t e, std::vector<Eigen::MatrixXf>& mats) {
  if(i > 0) {
    D.backward(i, D.Ys[i-1].transpose());       
  }
  else {
    D.backward(i, mats[e].middleRows(D.beg_row, D.batch_size).transpose());
  }
  //printf("B %d %d\n", i, e);
}

void omp_update_task(MNIST& D, size_t i) {
  D.update(i);
}


inline void fuck() {
  int a = 0;
  int b = 5;
  int g [10];
  int g2 [10];
  #pragma omp parallel 
  {
    #pragma omp single
    {
      for(int i=0; i<10; i++) {
        if(i == 0) {
          #pragma omp task depend(out: g[i], g2[i]) 
          printf("%d = %d\n", i , a++);
        }
        else {
          #pragma omp task depend(in: g[i-1], g2[i]) depend(out: g[i]) 
          printf("%d = %d\n", i , a++);
        }
      }
      for(int i=0; i<10; i++) {
        #pragma omp task depend(out: g2[i])
        b++;
      }

    }
  }
  exit(1);
  //std::cout << a << '/' << b << '\n';

}



inline void run_omp(MNIST& D) {
  // Create task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;
  //const auto iter_num = 2;

  // number of concurrent shuffle tasks
  const auto num_par_shf = std::min(D.num_storage, D.epoch);
  std::vector<Eigen::MatrixXf> mats(num_par_shf, D.images);
  std::vector<Eigen::VectorXi> vecs(num_par_shf, D.labels);

  const int num_layers = D.acts.size();

  // Propagation per epoch
  const auto prop_per_e = num_layers*iter_num;

  auto dep_s = new int [D.epoch];
  auto dep_f = new int [D.epoch * iter_num];
  auto dep_b = new int [D.epoch * prop_per_e];
  auto dep_u = new int [D.epoch * prop_per_e];


  // Each iteration has num_layers backward tasks 
  // Each epoch has iter_num iterations 
  // Total # of backward_tasks = iter_num * num_layers 
  #pragma omp parallel 
  {
    #pragma omp single
    {

       for(int e=0; e<D.epoch; e++) {
         // Shuffle Tasks
         if(e < num_par_shf) {
           #pragma omp task depend (out: dep_s[e])
           D.shuffle(mats[e%num_par_shf], vecs[e%num_par_shf], D.images.rows()); 
         }
         else {
           #pragma omp task depend (in: dep_s[e-num_par_shf], dep_b[(1+e-num_par_shf)*iter_num*num_layers-1]) depend (out: dep_s[e])
           D.shuffle(mats[e%num_par_shf], vecs[e%num_par_shf], D.images.rows());
         }

         // DNN operations
         for(int i=0; i<iter_num; i++) {
           // Forward tasks
           if(e == 0) {
             if(i == 0) {
               // The first task!!
               #pragma omp task depend (in: dep_s[e]) depend (out: dep_f[i])
               omp_forward_task(D, i, e%num_par_shf, mats, vecs);
             }
             else {
               // use openmp array sections syntax [lower_bound: length]
               #pragma omp task depend (in: dep_u[(i-1)*num_layers: num_layers]) depend (out: dep_f[i])
               omp_forward_task(D, i, e%num_par_shf, mats, vecs);
             }
           }
           else {
             if(i == 0) {
                //#pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e-1], dep_u[e*prop_per_e-2]) depend (out: dep_f[e*iter_num+i])
                #pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e-num_layers: num_layers]) depend (out: dep_f[e*iter_num+i])
                omp_forward_task(D, i, e%num_par_shf, mats, vecs);
             }
             else {
                //#pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers:num_layers]) depend (out: dep_f[e*iter_num+i])
                #pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers: num_layers]) depend (out: dep_f[e*iter_num+i])
                omp_forward_task(D, i, e%num_par_shf, mats, vecs);
             }
           }

           // Backward tasks   
           for(int j=num_layers-1; j>=0; j--) {
             if(j == num_layers-1) {
               #pragma omp task depend (in: dep_f[e*iter_num + i]) depend (out: dep_b[e*prop_per_e + i*num_layers + j])
               omp_backward_task(D, j, e%num_par_shf, mats);
             }
             else {
               #pragma omp task depend (in: dep_b[e*prop_per_e + i*num_layers + j + 1]) depend (out: dep_b[e*prop_per_e + i*num_layers + j])
               omp_backward_task(D, j, e%num_par_shf, mats);
             }
           }

           // Update tasks   
           for(int j=num_layers-1; j>=0; j--) {
             #pragma omp task depend (in: dep_b[e*prop_per_e + i*num_layers + j]) depend (out: dep_u[e*prop_per_e + i*num_layers + j])          
             omp_update_task(D, j);
           }


         }
       }
    } // End of omp single 
  } // End of omp parallel

  delete [] dep_s;
  delete [] dep_f;
  delete [] dep_b;
  delete [] dep_u;
}



inline void run_omp5(MNIST& D) {
  // Create task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;
  //const auto iter_num = 2;


  // number of concurrent shuffle tasks
  const auto num_par_shf = std::min(D.num_storage, D.epoch);
  std::vector<Eigen::MatrixXf> mats(num_par_shf, D.images);
  std::vector<Eigen::VectorXi> vecs(num_par_shf, D.labels);

  const int num_layers = D.acts.size();
  assert(num_layers == 2);

  // Propagation per epoch
  const auto prop_per_e = num_layers*iter_num;

  std::cout << "Here\n";
  auto dep_s = new int [D.epoch];
  auto dep_f = new int [D.epoch * iter_num];
  auto dep_b = new int [D.epoch * prop_per_e];
  auto dep_u = new int [D.epoch * prop_per_e];


  // Each iteration has num_layers backward tasks 
  // Each epoch has iter_num iterations 
  // Total # of backward_tasks = iter_num * num_layers 
  // #pragma omp parallel num_threads (1) 
  #pragma omp parallel 
  {
    #pragma omp single
    {

       for(int e=0; e<D.epoch; e++) {
         // Shuffle Tasks
         if(e < num_par_shf) {
           #pragma omp task depend (out: dep_s[e])
           D.shuffle(mats[e%num_par_shf], vecs[e%num_par_shf], D.images.rows()); 
           //printf("S %d\n", e);
         }
         else {
           #pragma omp task depend (in: dep_s[e-num_par_shf], dep_b[(1+e-num_par_shf)*iter_num*num_layers-1]) depend (out: dep_s[e])
           D.shuffle(mats[e%num_par_shf], vecs[e%num_par_shf], D.images.rows());
           //printf("S 2 %d\n", e);
         }

         // DNN operations
         for(int i=0; i<iter_num; i++) {
           // Forward tasks
           if(e == 0) {
             if(i == 0) {
               // The first task!!
               //#pragma omp task depend (in: dep_s[e]) depend (out: dep_f[i*num_layers])
               #pragma omp task depend (in: dep_s[e]) depend (out: dep_f[i])
               omp_forward_task(D, i, e%num_par_shf, mats, vecs);
             }
             else {
               // use openmp array sections syntax [lower_bound: length]
               //#pragma omp task depend (in: dep_u[(i-1)*num_layers: num_layers]) depend (out: dep_f[i])
               #pragma omp task depend (in: dep_u[(i-1)*num_layers], dep_u[(i-1)*num_layers+1]) depend (out: dep_f[i])
               omp_forward_task(D, i, e%num_par_shf, mats, vecs);
             }
           }
           else {
             if(i == 0) {
                //#pragma omp task depend (in: dep_s[e], dep_u[(e-1)*prop_per_e+(i-1)*num_layers]) depend (out: dep_f[e*prop_per_e + i*num_layers])
                //#pragma omp task depend (in: dep_s[e], dep_u[(e-1)*prop_per_e:num_layers]) depend (out: dep_f[e*iter_num+i])
                #pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e-1], dep_u[e*prop_per_e-2]) depend (out: dep_f[e*iter_num+i])
                omp_forward_task(D, i, e%num_par_shf, mats, vecs);
             }
             else {
 //               #pragma omp task depend (in: dep_u[(e-1)*prop_per_e+(i-1)*num_layers]) depend (out: dep_f[e*prop_per_e + i*num_layers])
                //#pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers:num_layers]) depend (out: dep_f[e*iter_num+i])
                #pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers], dep_u[e*prop_per_e+(i-1)*num_layers+1]) depend (out: dep_f[e*iter_num+i])
                omp_forward_task(D, i, e%num_par_shf, mats, vecs);
             }
           }

           // Backward tasks   
           for(int j=num_layers-1; j>=0; j--) {
             if(j == num_layers-1) {
               #pragma omp task depend (in: dep_f[e*iter_num + i]) depend (out: dep_b[e*prop_per_e + i*num_layers + j])
               omp_backward_task(D, j, e%num_par_shf, mats);
             }
             else {
               #pragma omp task depend (in: dep_b[e*prop_per_e + i*num_layers + j + 1]) depend (out: dep_b[e*prop_per_e + i*num_layers + j])
               omp_backward_task(D, j, e%num_par_shf, mats);
             }
           }

           // Update tasks   
           for(int j=num_layers-1; j>=0; j--) {
             #pragma omp task depend (in: dep_b[e*prop_per_e + i*num_layers + j]) depend (out: dep_u[e*prop_per_e + i*num_layers + j])          
             omp_update_task(D, j);
           }


         }
       }
    } // End of omp single 
  } // End of omp parallel

  delete [] dep_s;
  delete [] dep_f;
  delete [] dep_b;
  delete [] dep_u;
}





inline void run_omp4() {

  const int n = 10;
  const int iterations = 4;
  auto src = new int[n*n];
  auto dest = new int [n*n];
  for(size_t i=0; i<n*n; i++) {
    src[i] = 1;
    dest[i] = 1;
  }

  #pragma omp parallel
  {
    #pragma omp single
    {
      for(int i=0; i<iterations; i++) {
        for(int y=1; y<(n-1); ++y) {
          int x;
          #pragma omp task depend(in: src[x], src[x+n], src[x-n]) depend(out: dest[x])
          for(x=(y*n)+1; x<(y*n)+n-1; ++x) {
            dest[x]=(src[x-n]+src[x+n]+src[x]+src[x-1]+src[x+1]) *0.2;
          }
        }
        std::swap(dest, src);
      }
    }
  }


  for(size_t i=0; i<n*n; i++) {
    std::cout << dest[i] << ' ';
    if(i%n == n-1) std::cout << '\n';
    //assert(src[i] == 1);
    //assert(dest[i] == 1);
  }
}


inline void run_omp3() {
  int x[4];
  #pragma omp parallel
  {
    #pragma omp single
    {
      #pragma omp task depend( out: x[0] )
      x[0] = 1;
      #pragma omp task depend( in: x[0] ) depend( out: x[1])
      x[1] = x[0]*2;
      #pragma omp task depend( in: x[0] ) depend( out: x[2])
      x[2] = x[0]*3;
      #pragma omp task depend( in: x[1], x[2])
      x[3] = x[1] + x[2];
    }
  } 
  std::cout << x[0] << "/" << x[1] << "/" << x[2] << "/" << x[3] << '\n';
}



