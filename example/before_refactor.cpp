
#include <taskflow/taskflow.hpp>  // the only include you need
#include <cmath>
#include <experimental/filesystem>
#include <fstream>
#include <random>
#include <cmath>

#include <omp.h>
#include <Eigen/Dense>

// Function: read_mnist_label
auto read_mnist_label(const std::experimental::filesystem::path& path) {
  
  // Helper lambda.
	auto reverse_int = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i         & 255;
    c2 = (i >> 8)  & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };
  
  // Read the image.
  std::ifstream ifs(path, std::ios::binary);
  
  if(!ifs) {
    assert(false);
  }

  int magic_number = 0;
  int num_imgs = 0;

  ifs.read((char*)&magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);

  ifs.read((char*)&num_imgs, sizeof(num_imgs));
  num_imgs = reverse_int(num_imgs);
  
  Eigen::VectorXi labels(num_imgs);
  for (int i = 0; i<num_imgs; ++i) {
    unsigned char temp = 0;  // must use unsigned
    ifs.read((char*)&temp, sizeof(temp));
    labels[i] = static_cast<int>(temp);
  }
  return labels;
}


auto read_mnist_image(const std::experimental::filesystem::path& path) {
  
  // Helper lambda.
	auto reverse_int = [] (int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i         & 255;
    c2 = (i >> 8)  & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };
  
  // Read the image.
  std::ifstream ifs(path, std::ios::binary);

  if(!ifs) {
    assert(false);
  }

  int magic_number = 0;
  int num_imgs = 0;
  int num_rows = 0;
  int num_cols = 0;

  ifs.read((char*)&magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);

  ifs.read((char*)&num_imgs, sizeof(num_imgs));
  num_imgs = reverse_int(num_imgs);

  ifs.read((char*)&num_rows, sizeof(num_rows));
  num_rows = reverse_int(num_rows);

  ifs.read((char*)&num_cols, sizeof(num_cols));
  num_cols = reverse_int(num_cols);
 
  Eigen::MatrixXf images(num_imgs, num_rows*num_cols);

  for(int i = 0; i < num_imgs; ++i) {
    int j = 0;
    for(int r = 0; r < num_rows; ++r) {
      for(int c = 0; c < num_cols; ++c) {
        unsigned char p = 0;  // must use unsigned
        ifs.read((char*)&p, sizeof(p));
        images(i, r*num_cols + c) = static_cast<float>(p);
      }
    }
  }

  for(size_t i=0; i<images.rows(); i++) {
    for(size_t j=0; j<images.cols(); j++) {
      images(i, j) /= 255.0;
    }
  }
  return images;
}


auto time_diff(std::chrono::time_point<std::chrono::high_resolution_clock> &t1, 
               std::chrono::time_point<std::chrono::high_resolution_clock> &t2) {
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000000.0;
}



// ------------------------------------------------------------------------------------------------


enum class Activation {
  NONE,
  RELU,
  SIGMOID
};

enum class Optimizer {
  NONE,
  GradientDescent,
  Adam
};


// Procedure: sigmoid
inline void sigmoid(Eigen::MatrixXf& x) {
  x = ((1.0f + (-x).array().exp()).inverse()).matrix();
}

// Procedure: relu
inline void relu(Eigen::MatrixXf& x) {
  for(int j=0; j<x.cols(); ++j) {
    for(int i=0; i<x.rows(); ++i) {
      if(x(i, j) <= 0.0f) {
        x(i, j) = 0.0f;
      }   
    }   
  }
}


void activate(float &v, Activation act) {
  switch(act) {
    case Activation::NONE:    return;
    case Activation::RELU:    v = std::max(0.0f, v); return;
    case Activation::SIGMOID: v = 1.0/(1.0 + std::exp(-v)); return;
  };
}


void deactivate(float& v, Activation act) {
  switch(act) {
    case Activation::NONE:    v = 1; return; 
    case Activation::RELU:    v = std::max(0.0f, v); return;
    case Activation::SIGMOID: v = v*(1-v); return ;
  };
}

struct MNIST {

  struct Node {
    Node(size_t id, int layer, Activation act, MNIST &dnn): 
      id(id), layer(layer), act(act), dnn(dnn) { 
    }

    // X = 100 x 768, W = 768 x 30
    // Y = 100 x 30
    // Every node (768 x 1) calculates a column
    void forward() {
      if(layer == 0){
        dnn.Ys[layer].col(id) = (dnn.images.middleRows(dnn.beg_row, dnn.batch_size) * dnn.Ws[layer].col(id)).array()
          + dnn.Bs[layer](0, id);
      }
      else {
        dnn.Ys[layer].col(id) = (dnn.Ys[layer-1] * dnn.Ws[layer].col(id)).array()
                                + dnn.Bs[layer](0, id);
      }

      for(size_t i=0; i<dnn.Ys[layer].rows(); i++) {
        activate(dnn.Ys[layer](i, id) ,act);
      }
    }

    int layer;
    size_t id;
    Activation act;
    MNIST &dnn;
  };

  struct Layer {
    Layer(size_t in, size_t out, Activation act, size_t id, MNIST& dnn): in(in), out(out), act(act), id(id) {
      for(size_t i=0; i<out;i ++) {
        nodes.emplace_back(i, id, act, dnn);
      }
    }
    size_t id;
    size_t in;
    size_t out;
    Activation act;
  
    std::vector<Node> nodes;
    std::vector<tf::Task> tasks;

    std::vector<tf::Task> backward_tasks;
  };


  MNIST() {
    images = read_mnist_image("./train-images.data");
    labels = read_mnist_label("./train-labels.data");

    test_images = read_mnist_image("./t10k-images-idx3-ubyte");
    test_labels = read_mnist_label("./t10k-labels-idx1-ubyte");
  }

  auto& add_layer(size_t in_degree, size_t out_degree, Activation act) {
    layers.emplace_back(in_degree, out_degree, act, layers.size(), *this);
    //Ws.emplace_back().resize(in_degree, out_degree);
    //Bs.emplace_back().resize(1, out_degree);
    Ys.emplace_back().resize(batch_size, out_degree);
    Ws.push_back(Eigen::MatrixXf::Random(in_degree, out_degree));
    Bs.push_back(Eigen::MatrixXf::Random(1, out_degree));

    dW.emplace_back().resize(in_degree, out_degree);
    dB.emplace_back().resize(1, out_degree);
    D.emplace_back().resize(batch_size, out_degree);
    return layers.back();
  }

  void _forward_tasks() {
    for(size_t j=0; j<layers.size(); j++) {
      auto& l = layers[j];
      for(size_t i=0; i<l.nodes.size(); i++) {
        l.tasks.emplace_back(
          tf.silent_emplace(
            [&, i, j]() {
              layers[j].nodes[i].forward();
            }
          )
        ).name("I" + std::to_string(j) + "_N" + std::to_string(i));
      }

      //if(j > 0) {
      //  for(auto &t: l.tasks) {
      //    t.gather(layers[j-1].tasks);
      //  }
      //}

      if(j > 0) {
        sync_tasks.back().broadcast(layers[j].tasks);
      }
      if(j != layers.size() - 1) {
        auto& t = sync_tasks.emplace_back(tf.silent_emplace([](){}));
        t.gather(layers[j].tasks);
      }
    }

    input_layer = tf.silent_emplace(
      [&]() {
        //beg_row += batch_size;
        //if(beg_row >= images.rows()) {
        //  beg_row = 0;
        //}
      }
    ).name("Input");

    input_layer.broadcast(layers[0].tasks);
  }


  void infer() {
    Eigen::MatrixXf res = test_images; 
    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<layers.size(); i++) {
      res = res * Ws[i] + Bs[i].replicate(res.rows(), 1);
      if(layers[i].act == Activation::RELU) {
        relu(res);
      }
      else if(layers[i].act == Activation::SIGMOID) {
        sigmoid(res);
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Infer runtime: " << time_diff(t1, t2) << " s\n";

    size_t cnt {0};
    for(int k=0; k<res.rows(); k++) {
      int pred ; 
      res.row(k).maxCoeff(&pred);
      if(pred == test_labels[k]) {
        cnt ++;
      }
    }
    std::cout << "Accuracy: " << cnt << '/' << res.rows() << '\n';
  }

  void build_taskflow() {

    // Set up forward tasks first
    _forward_tasks();

    output_layer = tf.silent_emplace(
      [&, layers=&layers] () {

        delta = Ys.back();
        delta = (delta - delta.rowwise().maxCoeff().replicate(1, delta.cols())).array().exp().matrix();
        delta = delta.cwiseQuotient(delta.rowwise().sum().replicate(1, delta.cols()));

        if(beg_row == 0 && false) { 
          infer();
        }

        for(size_t i=beg_row, j=0; j<batch_size; i++, j++) {
          delta(j, labels[i]) -= 1.0;
        }
      }      
    ).name("Output");

    output_layer.gather(layers.back().tasks);

    // Back propagation
    for(int i=layers.size()-1; i>=0; i--) {
      auto &l = layers[i];
      for(size_t j=0; j<l.nodes.size(); j++) {
        if(i == layers.size() - 1) {
          l.backward_tasks.emplace_back(
            tf.silent_emplace(
              [&, i, j]() {
                dB[i](0, j) = 0.0;
                for(size_t k=0; k<Ys[i].rows(); k ++){
                  deactivate(Ys[i](k, j), l.act);
                  D[i](k, j) = delta(k, j) * Ys[i](k, j);
                  dB[i](0, j) += D[i](k, j);
                }
                dW[i].col(j) = Ys[i-1].transpose() * D[i].col(j);
                Bs[i](0, j) -= lrate*(dB[i](0, j) + decay*Bs[i](0, j)); 
              }
            )
          ).name("O" + std::to_string(i) + "_N" + std::to_string(j));
          output_layer.precede(l.backward_tasks.back());
        }
        else { 
          l.backward_tasks.emplace_back(
            tf.silent_emplace(
              [&, i, j]() {

                dB[i](0, j) = 0.0;
                D[i].col(j) = D[i+1] * Ws[i+1].row(j).transpose(); 
                // Update weight
                Ws[i+1].row(j) -= lrate*(dW[i+1].row(j) + decay*Ws[i+1].row(j));
                for(size_t k=0; k<Ys[i].rows(); k ++){
                  deactivate(Ys[i](k, j), l.act);
                  D[i](k, j) = D[i](k, j) * Ys[i](k, j);
                  dB[i](0, j) += D[i](k, j);
                }
                Bs[i](0, j) -= lrate*(dB[i](0, j) + decay*Bs[i](0, j)); 

                if(i > 0) {
                  dW[i].col(j) = Ys[i-1].transpose() * D[i].col(j);
                }
                else {
                  dW[i].col(j) = images.middleRows(beg_row, batch_size).transpose() * D[i].col(j);
                  Ws[i].col(j) -= lrate*(dW[i].col(j) + decay*Ws[i].col(j));
                }
              }
            )
          ).name("O" + std::to_string(i) + "_N" + std::to_string(j));
        }
      }
    }

    // Build precedence
    //for(size_t i=layers.size()-1; i>0; i--) {
    //  for(auto& t: layers[i].backward_tasks) {
    //    t.broadcast(layers[i-1].backward_tasks);
    //  }
    //}
    for(size_t i=layers.size()-1; i>0; i--) {
      auto& t = sync_tasks.emplace_back(tf.silent_emplace([](){}));
      t.gather(layers[i].backward_tasks);
      t.broadcast(layers[i-1].backward_tasks);
    }
  }


  void parallel_matrix() {
    std::cout << "Parallel matrix version\n";
    build_taskflow();
    //std::cout << tf.dump() << '\n';
    //exit(1);

    std::random_device rd;
    std::mt19937 gen(rd());
    const auto iter_num = images.rows()/batch_size;
    std::cout << "Start training...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    for(auto i=0; i<epoch; i++) { 
      for(auto j=0; j<iter_num; j++) {
        tf.wait_for_all();  // block until finished  
        beg_row += batch_size;
        if(beg_row >= images.rows()) {
          beg_row = 0;
        }
      }

      {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p(images.rows());
        p.setIdentity();
        std::shuffle(p.indices().data(), p.indices().data() + p.indices().size(), gen);
        images = p * images;
        labels = p * labels;
        beg_row = 0;
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel matrix runtime: " << time_diff(t1, t2) << " s\n";
    infer();
  }


  void seqential() {
    std::cout << "Seqential version\n";
    std::random_device rd;
    std::mt19937 gen(rd());
    const auto iter_num = images.rows()/batch_size;

    //omp_set_num_threads(4);
    //Eigen::setNbThreads(4);
    //std::cout << Eigen::nbThreads() << "\n";

    std::cout << "Start training..\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    for(auto e=0; e<epoch; e++) { 
      //std::cout << e << ' ';
      for(auto it=0; it<iter_num; it++) {
        // Foward propagation
        for(size_t i=0; i<layers.size(); i++) {
          if(i == 0){
            Ys[i] = images.middleRows(beg_row, batch_size) * Ws[i] + Bs[i].replicate(batch_size, 1);
          }
          else {
            Ys[i] = Ys[i-1] * Ws[i] + Bs[i].replicate(Ys[i-1].rows(), 1);
          }

          for(size_t j=0; j<Ys[i].rows(); j++) {
            for(size_t k=0; k<Ys[i].cols(); k++) {
              activate(Ys[i](j, k), layers[i].act);
            }
          }
        }

        // Loss 
        delta = Ys.back();
        delta = (delta - delta.rowwise().maxCoeff().replicate(1, delta.cols())).array().exp().matrix();
        delta = delta.cwiseQuotient(delta.rowwise().sum().replicate(1, delta.cols()));

        if(beg_row == 0 && false) { 
          infer();
        }

        for(size_t i=beg_row, j=0; j<batch_size; i++, j++) {
          delta(j, labels[i]) -= 1.0;
        }

        // Backward propagation
        for(int i=layers.size()-1; i>=0; i--) {
          auto &l = layers[i];
          for(size_t j=0; j<Ys[i].rows();  j++) {
            for(size_t k=0; k<Ys[i].cols(); k++) {
              deactivate(Ys[i](j, k), l.act);
            }
          }
          delta = delta.cwiseProduct(Ys[i]);

          dB[i] = delta.colwise().sum();
          if(i > 0) {
            dW[i] = Ys[i-1].transpose() * delta;
          }
          else {
            dW[i] = images.middleRows(beg_row, batch_size).transpose() * delta;
          }

          if(i > 0) {
            delta = delta * Ws[i].transpose();
          }
        }

        // Update parameters
        for(int i=layers.size()-1; i>=0; i--) {
          Ws[i] -= lrate*(dW[i] + decay*Ws[i]);
          Bs[i] -= lrate*(dB[i] + decay*Bs[i]); 
        }

        beg_row += batch_size;
        if(beg_row >= images.rows()) {
          beg_row = 0;
        }
      }

      // Shuffle input
      {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p(images.rows());
        p.setIdentity();
        std::shuffle(p.indices().data(), p.indices().data() + p.indices().size(), gen);
        images = p * images;
        labels = p * labels;
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Seqential runtime: " << time_diff(t1, t2) << " s\n";
    infer();
  }


  void rand_perm(Eigen::MatrixXf& mat, Eigen::VectorXi& vec, const size_t row_num) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p(row_num);
    p.setIdentity();
    std::shuffle(p.indices().data(), p.indices().data() + p.indices().size(), gen);

    mat = p * mat;
    vec = p * vec;
  }

  void pipeline() {
    std::cout << "Pipeline version\n";
    std::vector<tf::Task> forward_tasks;
    std::vector<tf::Task> backward_tasks;
    std::vector<tf::Task> update_tasks;

    const auto num_storage = 16;

    std::vector<tf::Task> shuffle_tasks;
    std::vector<Eigen::MatrixXf> mats(num_storage, images);
    std::vector<Eigen::VectorXi> vecs(num_storage, labels);
 
    // Create task flow graph
    const auto iter_num = images.rows()/batch_size;

    for(auto e=0; e<epoch; e++) {
      for(auto i=0; i<iter_num; i++) {
        auto& f_task = forward_tasks.emplace_back(
          tf.silent_emplace(
            [&, iter = i, e=e%num_storage]() {
              if(iter == 0 && false) infer();

              if(iter != 0) {
                beg_row += batch_size;
                if(beg_row >= images.rows()) {
                  beg_row = 0;
                }
              }
              for(size_t i=0; i<layers.size(); i++) {
                if(i == 0){
                  Ys[i] = mats[e].middleRows(beg_row, batch_size) * Ws[i] + Bs[i].replicate(batch_size, 1);
                }
                else {
                  Ys[i] = Ys[i-1] * Ws[i] + Bs[i].replicate(Ys[i-1].rows(), 1);
                }

                for(size_t j=0; j<Ys[i].rows(); j++) {
                  for(size_t k=0; k<Ys[i].cols(); k++) {
                    activate(Ys[i](j, k), layers[i].act);
                  }
                }
              }
              // Loss 
              delta = Ys.back();
              delta = (delta - delta.rowwise().maxCoeff().replicate(1, delta.cols())).array().exp().matrix();
              delta = delta.cwiseQuotient(delta.rowwise().sum().replicate(1, delta.cols()));

              for(size_t i=beg_row, j=0; j<batch_size; i++, j++) {
                delta(j, vecs[e][i]) -= 1.0;
              }
            }
          ) 
        ).name("e" + std::to_string(e) + "_F"+std::to_string(i));

        if(i == 0 && e != 0) {
          auto sz = update_tasks.size();
          for(auto j=1; j<=layers.size() ;j++) {
            update_tasks[sz-j].precede(f_task);
          }         
        }

        if(i != 0) {
          auto sz = update_tasks.size();
          for(auto j=1; j<=layers.size() ;j++) {
            update_tasks[sz-j].precede(f_task);
          }
        }

        for(int j=layers.size()-1; j>=0; j--) {
          // backward propagation
          auto& b_task = backward_tasks.emplace_back(
            tf.silent_emplace(
              [&, i=j, e=e%num_storage] () {
                auto &l = layers[i];
                for(size_t j=0; j<Ys[i].rows();  j++) {
                  for(size_t k=0; k<Ys[i].cols(); k++) {
                    deactivate(Ys[i](j, k), l.act);
                  }
                }
                delta = delta.cwiseProduct(Ys[i]);
      
                dB[i] = delta.colwise().sum();
                if(i > 0) {
                  dW[i] = Ys[i-1].transpose() * delta;
                }
                else {
                  dW[i] = mats[e].middleRows(beg_row, batch_size).transpose() * delta;
                }
      
                if(i > 0) {
                  delta = delta * Ws[i].transpose();
                }
              }
            )
          ).name("e" + std::to_string(e) + "_L" + std::to_string(i) +"_B" + std::to_string(j));
          // update weight 
          auto& u_task = update_tasks.emplace_back(
            tf.silent_emplace(
              [&, i=j] () {
                Ws[i] -= lrate*(dW[i] + decay*Ws[i]);
                Bs[i] -= lrate*(dB[i] + decay*Bs[i]);
              }
            )
          ).name("e" + std::to_string(e) + "_L" + std::to_string(i) +"_U" + std::to_string(j));

          if(j == layers.size() - 1) {
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
                     .name("e" + std::to_string(e) + "_S" + std::to_string(e%num_storage));
      }
      else {
        auto& t = shuffle_tasks.emplace_back(
          tf.silent_emplace(
          [&, e=e%num_storage]() {
            rand_perm(mats[e], vecs[e], images.rows());
          })
        ).precede(forward_tasks[forward_tasks.size()-iter_num])           
         .name("e" + std::to_string(e) + "_S" + std::to_string(e%num_storage));

        if(e >= num_storage) {
          shuffle_tasks[e-num_storage].precede(t);
        }
      }
    } // End of all epoch

    std::cout << "Start training...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    tf.wait_for_all();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Pipeline runtime: " << time_diff(t1, t2) << " s\n";
    infer();
  }


  // Parameter functions --------------------------------------------------------------------------
  auto& epoch_num(size_t e) {
    epoch = e;
    return *this;
  }
  auto& batch(size_t b) {
    batch_size = b;
    assert(images.rows()%batch_size == 0);
    return *this;
  }
  auto& learning_rate(float lrate) {
    lrate = lrate;
    return *this;
  }

  std::vector<Eigen::MatrixXf> Ys;
  std::vector<Eigen::MatrixXf> Ws;
  std::vector<Eigen::MatrixXf> Bs;

  std::vector<Eigen::MatrixXf> dW;
  std::vector<Eigen::MatrixXf> dB;

  std::vector<Eigen::MatrixXf> D;

  std::vector<Layer> layers;
  std::vector<tf::Task> sync_tasks;

  // Training images # = 60000 x 784 (28 x 28)
  Eigen::MatrixXf images;
  Eigen::VectorXi labels;
  Eigen::MatrixXf delta;

  // Testing images # = 10000 x 784 (28 x 28)
  Eigen::MatrixXf test_images;
  Eigen::VectorXi test_labels;

  size_t beg_row {0};

  tf::Task input_layer;
  tf::Task output_layer;

  double lrate {0.01};
  double decay {0.01};

  size_t epoch {0};
  size_t batch_size {1};

  tf::Taskflow tf {4};
};

int main(int argc, char *argv[]){
  
  MNIST dnn;
  dnn.epoch_num(50).batch(100).learning_rate(0.001);
  dnn.add_layer(784, 100, Activation::RELU);
  dnn.add_layer(100, 100, Activation::RELU);
  dnn.add_layer(100, 100, Activation::RELU);
  dnn.add_layer(100, 100, Activation::RELU);
  dnn.add_layer(100, 10, Activation::NONE); 

  int sel = 0;
  if(argc > 1) {
    if(::strcmp(argv[1], "pipeline") == 0) {
      sel = 1;
    }
    else if(::strcmp(argv[1], "matrix") == 0) {
      sel = 2;
    }
  }

  switch(sel) {
    case 1:  dnn.pipeline(); break;
    case 2:  dnn.parallel_matrix(); break;
    default: dnn.seqential(); break;
  };

  return 0;
}



