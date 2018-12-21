#include <fstream>
#include <random>
#include <cmath>
#include <iostream>

#include <Eigen/Dense>

void test(Eigen::MatrixXf &&mat) {
  std::cout << mat.rows() << " " << mat.cols() << "\n";
  mat(0, 0) = 100.0;
}

void test(float &v) {
  v = 100.0;
}

int main() {
  Eigen::MatrixXf m(4,4);
  m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;

  //test(m.col(0));
  test(m(0, 0));
  std::cout << m << "\n";
  exit(1);

  Eigen::MatrixXf m2(4,4);
  m2 <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;


  auto m3 = m*m2.col(0);
  std::cout << m3 << '\n';
}
