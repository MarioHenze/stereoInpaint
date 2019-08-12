#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

class CostVolume {
public:
  CostVolume(cv::Rect const &size, size_t const d);
  void calculate(cv::Mat const &left, cv::Mat const &right,
                 cv::Mat const &mask = cv::Mat(), size_t block_size = 5);

private:
  // vector of scanlines, of pixels, of matching cost
  std::vector<size_t> m_cost_volume;
  const size_t m_max_displacement;

  /**
   * @brief to_linear maps a cost volume coordinate into linear memory
   * @param x horizontal position
   * @param y vertical position
   * @param d displacement
   * @return the linear index into memory
   */
  size_t to_linear(size_t x, size_t y, size_t d);
};
