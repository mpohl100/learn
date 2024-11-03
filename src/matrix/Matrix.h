#pragma once

#include <stdexcept>
#include <vector>

namespace matrix {

template <class T> class Matrix {
public:
  Matrix(const size_t rows, const size_t cols) {
    _rows = rows;
    _cols = cols;
    _data = std::vector<T>(_rows * _cols, T{});
  }

  Matrix() = default;
  Matrix(const Matrix &) = default;
  Matrix(Matrix &&) = default;
  Matrix &operator=(const Matrix &) = default;
  Matrix &operator=(Matrix &&) = default;

  T &get(size_t x, size_t y) {
    if (x >= _rows || y >= _cols)
      throw std::out_of_range(
          "Matrix::get out of range x: " + std::to_string(x) +
          " y: " + std::to_string(y) + " rows: " + std::to_string(_rows) +
          " cols: " + std::to_string(_cols));
    return _data[x * _rows + y];
  }
  const T &get(size_t x, size_t y) const {
    if (x >= _rows || y >= _cols)
      throw std::out_of_range(
          "Matrix::get out of range x: " + std::to_string(x) +
          " y: " + std::to_string(y) + " rows: " + std::to_string(_rows) +
          " cols: " + std::to_string(_cols));
    return _data[x * _rows + y];
  }

  size_t rows() const { return _rows; }

  size_t cols() const { return _cols; }

private:
  size_t _rows = 1;
  size_t _cols = 1;
  std::vector<T> _data = {T{}};
};

} // namespace matrix