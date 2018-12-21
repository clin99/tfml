
#include <taskflow/taskflow.hpp>  // the only include you need
#include <cmath>
#include <experimental/filesystem>
#include <fstream>
#include <random>
#include <cmath>

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
        );
      }
      if(j > 0) {
        for(auto &t: l.tasks) {
          t.gather(layers[j-1].tasks);
        }
      }
    }

    input_layer = tf.silent_emplace(
      [&]() {
        beg_row += batch_size;
        if(beg_row >= images.rows()) {
          beg_row = 0;
        }
      }
    );

    input_layer.broadcast(layers[0].tasks);
  }

  // https://github.com/twhuang-uiuc/DtCraft/blob/master/src/ml/dnn.cpp
  void build_taskflow() {

    // Set up forward tasks first
    _forward_tasks();

    output_layer = tf.silent_emplace(
      [&, layers=&layers] () {

        delta = Ys.back();
        delta = (delta - delta.rowwise().maxCoeff().replicate(1, delta.cols())).array().exp().matrix();
        delta = delta.cwiseQuotient(delta.rowwise().sum().replicate(1, delta.cols()));

        Eigen::VectorXi label(delta.rows());

        for(int k=0; k<delta.rows(); ++k) {
          //std::cout << delta.row(k) << '\n';
          delta.row(k).maxCoeff(&label(k));
        }

        if(beg_row == 0) {
          size_t cnt {0};
          for(int k=0; k<label.rows(); k++) {
            if(label[k] == labels[k+beg_row]) {
              cnt ++;
            }
          }
          std::cout << "correct: " << cnt << '\n';
        }

        for(size_t i=beg_row, j=0; j<batch_size; i++, j++) {
          delta(j, labels[i]) -= 1.0;
        }

        //if(epoch%100 == 0) {
        //  if(pred == labels[cur_img])
        //    std::cout << "\033[0;31m" << pred << ' ' << labels[cur_img] << "\033[0m" << '\n'; 
        //  else
        //    std::cout << pred << ' ' << labels[cur_img] << '\n'; 
        //  epoch = 1;
        //}
        //else {
        //  epoch ++;
        //}
      }      
    );

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
                Bs[i](0, j) -= lrate*(dB[i](0, j) + decay*dB[i](0, j)); 
              }
            )
          );
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
                Bs[i](0, j) -= lrate*(dB[i](0, j) + decay*dB[i](0, j)); 

                if(i > 0) {
                  dW[i].col(j) = Ys[i-1].transpose() * D[i].col(j);
                }
                else {
                  dW[i].col(j) = images.middleRows(beg_row, batch_size).transpose() * D[i].col(j);
                  Ws[i].col(j) -= lrate*(dW[i].col(j) + decay*Ws[i].col(j));
                }
              }
            )
          );
        }
      }
    }

    // Build precedence
    for(size_t i=layers.size()-1; i>0; i--) {
      for(auto& t: layers[i].backward_tasks) {
        t.broadcast(layers[i-1].backward_tasks);
      }
    }

  }


  void run() {
    build_taskflow();

    std::random_device rd;
    std::mt19937 gen(rd());
    auto iter_num = images.rows()/batch_size;
    for(auto i=0; i<epoch; i++) { 

      std::cout << i << ' ';
      auto t1 = std::chrono::high_resolution_clock::now();
      for(auto j=0; j<iter_num; j++) {
        tf.wait_for_all();  // block until finished 
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      std::cout << diff << " us\n";

      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p(images.rows());
      p.setIdentity();
      std::shuffle(p.indices().data(), p.indices().data() + p.indices().size(), gen);
      images = p * images;
      labels = p * labels;
      beg_row = 0;
    }
  }


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
  Eigen::MatrixXf images;
  Eigen::VectorXi labels;
  Eigen::MatrixXf delta;

  size_t beg_row {0};

  tf::Task input_layer;
  tf::Task output_layer;

  double lrate {0.01};
  double decay {0.01};

  size_t epoch {0};
  size_t batch_size {1};

  tf::Taskflow tf {1};
};

int main(){
  
  ::srand(1);


  MNIST dnn;
  dnn.epoch_num(30).batch(100).learning_rate(0.001);
  dnn.add_layer(784, 30, Activation::RELU);
  dnn.add_layer(30, 10, Activation::NONE); 

  //dnn.add_layer(784, 100, Activation::RELU);
  //dnn.add_layer(100, 30, Activation::SIGMOID);
  //dnn.add_layer(30, 20, Activation::NONE);
  //dnn.add_layer(20, 10, Activation::NONE);

  dnn.run();


  // Training images # = 60000 

  return 0;
}



