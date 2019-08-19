#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

bool operator==(cv::Point2i const &a, cv::Point2i const &b);

class CostVolume
{
public:
    CostVolume(const cv::Rect &size, const size_t d);
    void calculate(cv::Mat const &left,
                   cv::Mat const &right,
                   cv::Mat const &left_mask = cv::Mat(),
                   cv::Mat const &right_mask = cv::Mat(),
                   size_t const block_size = 5);

    /**
     * @brief slice draws a slice of the cost volume as image
     * @param y the scanline at which to slice
     * @return the disparity slice
     */
    cv::Mat slice(const size_t y) const;

    bool is_valid() const;

private:
    //! vector of scanlines, of pixels, of matching cost
    std::vector<size_t> m_cost_volume;

    //! If a pixel is contained in this mask all disparity values must be
    //! guessed in accordance to their neighbours
    cv::Mat m_mask;

    //! Holds the maximum displacement in both directions
    const size_t m_max_displacement;

    //! Holds the frontal facing size of the cost volume (i.e. width and height)
    cv::Rect m_size;

    /**
     * @brief to_linear maps a cost volume coordinate into linear memory
     * @param scanline vertical position
     * @param x horizontal position
     * @param d displacement
     * @param scanline_dim scanline count
     * @param d_dim disparity count
     * @return the linear index into memory
     */
    size_t to_linear(const size_t scanline,
                     const size_t x,
                     const size_t d,
                     const size_t scanline_dim,
                     const size_t x_dim) const;
    /**
     * @brief is_masked returns if a pixel location in the cost volume is masked
     * @param pixel the location in the cost volume
     * @return true if the location is masked, false otherwise
     */
    bool is_masked(cv::Point2i const &pixel) const;
};
