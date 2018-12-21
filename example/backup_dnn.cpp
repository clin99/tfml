
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
  
  std::vector<int> labels(num_imgs);
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
 
  std::vector<std::vector<double>> images;
  images.resize(num_imgs);
  for(auto& img: images) {
    img.resize(num_rows * num_cols);
  }

  for(int i = 0; i < num_imgs; ++i) {
    int j = 0;
    for(int r = 0; r < num_rows; ++r) {
      for(int c = 0; c < num_cols; ++c) {
        unsigned char p = 0;  // must use unsigned
        ifs.read((char*)&p, sizeof(p));
        images[i][r*num_cols + c] = static_cast<double>(p);
      }
    }
  }

  for(auto& img: images){
    for(auto &p: img) p /= 255.0;
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

double activate(double val, Activation act) {
  switch(act) {
    case Activation::NONE:    return val;
    case Activation::RELU:    return std::max(0.0, val);
    case Activation::SIGMOID: return 1.0/(1.0 + std::exp(-val));
  };
  assert(false);
  return -1.0;
}

double deactivate(double val, Activation act) {
  switch(act) {
    case Activation::NONE:    return 1.0;
    case Activation::RELU:    return val > 0.0 ? 1.0 : 0.0;
    case Activation::SIGMOID: return val * (1.0 - val);
  };
  assert(false);
  return -1.0;
}


struct MNIST {

  struct Node {
    Node(size_t in, int layer, Activation act, MNIST &dnn): layer(layer), act(act), dnn(dnn){ 
      weights.resize(in); 
      dW.resize(in);
      for(auto&w : weights) {
        w = ((double) rand() / (RAND_MAX));
      }
    }

    void forward() {
      y = 0.0;

      if(layer == 0){
        for(size_t i=0; i<weights.size(); i++) {
          y += dnn.images[dnn.cur_img][i] * weights[i] + bias;
        }
      }
      else {
        for(size_t i=0; i<weights.size(); i++) {
          y += dnn.layers[layer-1].nodes[i].y * weights[i] + bias;
        }
      }

      y = activate(y ,act);
    }

    std::vector<double> weights;
    std::vector<double> dW;
    double bias {0.0};
    double dB;

    double delta;
    double y {0.0};
    int layer;
    Activation act;
    MNIST &dnn;
  };

  struct Layer {
    Layer(size_t in, size_t out, Activation act, size_t id, MNIST& dnn): in(in), out(out), act(act), id(id) {
      for(size_t i=0; i<out;i ++) {
        nodes.emplace_back(in, id, act, dnn);
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

    // The output/prediction size is 10
    delta.resize(10); 
  }


  auto& add_layer(size_t in_degree, size_t out_degree, Activation act) {
    layers.emplace_back(in_degree, out_degree, act, layers.size(), *this);
    return layers.back();
  }

  void softmax(std::vector<double>& values) {
    auto max = *std::max_element(values.begin(), values.end());
    double sum {0.0};
    std::for_each(values.begin(), values.end(), [&max](auto& v){ v-= max; });
    std::for_each(values.begin(), values.end(), [&sum](auto& v){ v = std::exp(v); sum += v; }); 
    std::for_each(values.begin(), values.end(), [&sum](auto& v){ v /= sum; }); 
  }



  void update(double &old, double &delta) const {
    old -= lrate*(delta + decay*old);
  }

  // https://github.com/twhuang-uiuc/DtCraft/blob/master/src/ml/dnn.cpp
  void last_layer(tf::Taskflow& tf) {

    output_layer = tf.silent_emplace(
      [&, layers=&layers] () {
        auto &output_nodes = (*layers)[layers->size()-1].nodes; 

        assert(delta.size() == output_nodes.size());
        for(size_t i=0; i<output_nodes.size(); i++) {
          delta[i] = output_nodes[i].y;
        }

        auto pred = std::distance(delta.begin(), std::max_element(delta.begin(), delta.end()));

        // Calculate softmax loss
        softmax(delta);
        //exit(1);

        // Calculate dloss of softmax 
        delta[labels[cur_img]] -= 1.0;

        //for(auto &d: delta) {
        //  std::cout << d << ' ';
        //}
        //std::cout << '\n';

        if(epoch%100 == 0) {
          if(pred == labels[cur_img])
            std::cout << "\033[0;31m" << pred << ' ' << labels[cur_img] << "\033[0m" << '\n'; 
          else
            std::cout << pred << ' ' << labels[cur_img] << '\n'; 
          epoch = 1;
        }
        else {
          epoch ++;
        }
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
                auto &l = layers[i];
                l.nodes[j].delta = delta[j] * deactivate(l.nodes[j].y, l.act);
                l.nodes[j].dB = l.nodes[j].delta;
                update(l.nodes[j].bias, l.nodes[j].dB);

                for(size_t k=0; k<l.nodes[j].dW.size(); k++) {
                  l.nodes[j].dW[k] = l.nodes[j].delta * layers[i-1].nodes[k].y;
                }
          
              }
            )
          );
          output_layer.precede(l.backward_tasks.back());
        }
        else { 
          l.backward_tasks.emplace_back(
            tf.silent_emplace(
              [&, i, j]() {
                auto &l = layers[i];
                double delta = 0.0;
                for(size_t k=0; k<layers[i+1].out; k++) { 
                  delta += layers[i+1].nodes[k].delta * layers[i+1].nodes[k].weights[j];
                  // Update the weights of preceding layer
                  update(layers[i+1].nodes[k].weights[j], layers[i+1].nodes[k].dW[j]);
                }
                l.nodes[j].delta = delta * deactivate(l.nodes[j].y, l.act);
                l.nodes[j].dB = delta;
                update(l.nodes[j].bias, l.nodes[j].dB);

                for(size_t k=0; k<l.nodes[j].dW.size(); k++) {
                  if(i > 0) {
                    l.nodes[j].dW[k] = l.nodes[j].delta * layers[i-1].nodes[k].y;
                  }
                  else {
                    l.nodes[j].dW[k] = l.nodes[j].delta * images[cur_img][k];
                    update(l.nodes[j].weights[k], l.nodes[j].dW[k]);
                  }
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

  std::vector<Layer> layers;
  std::vector<std::vector<double>> images;
  std::vector<int> labels;
  size_t cur_img {0};

  tf::Task output_layer;
  std::vector<double> delta;

  double lrate {0.01};
  double decay {0.01};

  size_t epoch {0};
  size_t batch_size {1};
};

int main(){
  
  ::srand(1);

  tf::Taskflow tf(4);
  MNIST dnn;
  dnn.add_layer(784, 30, Activation::RELU);
  dnn.add_layer(30, 10, Activation::NONE);
 
  for(size_t j=0; j<dnn.layers.size(); j++) {
    auto& l = dnn.layers[j];
    for(size_t i=0; i<l.nodes.size(); i++) {
      l.tasks.emplace_back(
        tf.silent_emplace(
          [&, i, j]() {
            dnn.layers[j].nodes[i].forward();
          }
        )
      );
    }
    if(j > 0) {
      for(auto &t: l.tasks) {
        t.gather(dnn.layers[j-1].tasks);
      }
    }
  }


  dnn.last_layer(tf);


  //for(size_t i=0; i < dnn.images.size(); i++) {
  for(size_t i=0; i<10000000; i++) {
    //std::cout << "Start " << i << "  ";
    dnn.cur_img = i%dnn.images.size();
    //std::cout << tf.dump() << '\n';
    tf.wait_for_all();  // block until finished 
  }

  return 0;
}



