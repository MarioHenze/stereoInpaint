#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

class CostVolume
{
public:
    CostVolume(cv::Rect const &size, size_t const d);
    void calculate(cv::Mat const &left,
                   cv::Mat const &right,
                   cv::Mat const &mask = cv::Mat(),
                   size_t block_size = 5);

    /**
   * @brief slice draws a slice of the cost volume as image
   * @param y the scanline at which to slice
   * @return the disparity slice
   */
    cv::Mat slice(const size_t y) const;

    bool is_valid() const;

private:
    // vector of scanlines, of pixels, of matching cost
    std::vector<size_t> m_cost_volume;
    const size_t m_max_displacement;
    cv::Rect m_size;

    /**
   * @brief to_linear maps a cost volume coordinate into linear memory
   * @param x horizontal position
   * @param scanline vertical position
   * @param d displacement
   * @param scanline_dim scanline count
   * @param d_dim disparity count
   * @return the linear index into memory
   */
    size_t to_linear(const size_t x,
                     const size_t scanline,
                     const size_t d,
                     const size_t scanline_dim,
                     const size_t d_dim) const;
};
