#pragma once

#include <cstddef>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

template<typename U, typename T>
U narrow(T const big);

class CostVolume
{
public:
    CostVolume(const size_t height, const size_t width, const size_t d);
    void calculate(cv::Mat const &left,
                   cv::Mat const &right,
                   cv::Mat const &left_mask = cv::Mat(),
                   cv::Mat const &right_mask = cv::Mat(),
                   size_t const blocksize = 5);

    /**
     * @brief slice draws a slice of the cost volume as image
     * @param scanline the scanline at which to slice
     * @return the disparity slice
     */
    cv::Mat slice(const size_t scanline) const;

    /**
     * @brief slice_count reports the number of possible slices
     * @return the slice count of the cost volume
     */
    size_t slice_count() const;

    bool is_valid() const;

private:
    //! vector of scanlines, of pixels, of matching cost
    std::vector<size_t> m_cost_volume;

    //! If a pixel is contained in this mask all disparity values must be
    //! guessed in accordance to their neighbours
    cv::Mat m_mask;

    //! The height of the volume
    size_t const m_scanline_count;

    //! The amount of pixels on one scanline
    size_t const m_pixels_per_scanline;

    //! The amount of displacements of one pixel
    size_t const m_displacements_per_pixel;

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
                     const size_t d) const;
    /**
     * @brief is_masked returns if a pixel location in the cost volume is masked
     * @param pixel the location in the cost volume
     * @return true if the location is masked, false otherwise
     */
    bool is_masked(cv::Point2i const &pixel) const;
};
