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
    CostVolume(const int height, const int width, const int d);
    void calculate(cv::Mat const &left,
                   cv::Mat const &right,
                   cv::Mat const &left_mask = cv::Mat(),
                   cv::Mat const &right_mask = cv::Mat(),
                   int const blocksize = 5);

    /**
     * @brief slice draws a slice of the cost volume as image
     * @param scanline the scanline at which to slice
     * @return the disparity slice
     */
    cv::Mat slice(const int scanline) const;

    /**
     * @brief slice_count reports the number of possible slices
     * @return the slice count of the cost volume
     */
    int slice_count() const;

    bool is_valid() const;

private:
    //! vector of scanlines, of pixels, of matching cost
    std::vector<int> m_cost_volume;

    //! If a pixel is contained in this mask all disparity values must be
    //! guessed in accordance to their neighbours
    cv::Mat m_mask;

    //! The height of the volume
    int const m_scanline_count;

    //! The amount of pixels on one scanline
    int const m_pixels_per_scanline;

    //! The amount of displacements of one pixel
    int const m_displacements_per_pixel;

    /**
     * @brief to_linear maps a cost volume coordinate into linear memory
     * @param scanline vertical position
     * @param x horizontal position
     * @param d displacement
     * @param scanline_dim scanline count
     * @param d_dim disparity count
     * @return the linear index into memory
     */
    size_t to_linear(const int scanline, const int x, const int d) const;
    /**
     * @brief is_masked returns if a pixel location in the cost volume is masked
     * @param pixel the location in the cost volume
     * @return true if the location is masked, false otherwise
     */
    bool is_masked(cv::Point2i const &pixel) const;
};
