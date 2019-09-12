#include "costvolume.h"

#include <limits>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <forward_list>
#include <type_traits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

template<typename U, typename T>
U narrow(T const big)
{
    static_assert(std::is_integral<T>::value, "Type T needs to be integral");
    static_assert(std::is_unsigned<T>::value, "Type T needs to be unsigned");
    static_assert(std::is_integral<U>::value, "Type U needs to be integral");

    assert(std::numeric_limits<U>::max() >= big);
    return static_cast<U>(big);
}

cv::Rect clamp_into(cv::Rect const & r, cv::Rect const & region)
{
    cv::Point2i const upper_left(
        std::clamp(r.x, region.x, region.x + region.width),
        std::clamp(r.y, region.y, region.y + region.height));

    cv::Point2i const lower_right(
        std::clamp(r.x + r.width, region.x, region.x + region.width),
        std::clamp(r.y + r.height, region.y, region.y + region.height));

    return cv::Rect(upper_left, lower_right);
}

cv::Rect clamp_into(cv::Rect const & r, cv::Mat const & region)
{
    return clamp_into(r, cv::Rect(0,0,region.cols, region.rows));
}

cv::Rect overlap(std::forward_list<cv::Rect> rects)
{
    cv::Rect region = rects.front();
    std::for_each(rects.cbegin(),
                  rects.cend(),
                  [&region](cv::Rect r) { clamp_into(region, r); });

    return region;
}

CostVolume::CostVolume(int const width, int const height, int const d)
    : m_cost_volume(static_cast<size_t>(width * height * (2 * d + 1)), 0)
    , m_scanline_count(height)
    , m_pixels_per_scanline(width)
    , m_displacements_per_pixel(2 * d + 1)
{
    assert(height > 0);
    assert(width > 0);
    assert(d > 0);
}

void CostVolume::calculate(cv::Mat const &left,
                           cv::Mat const &right,
                           cv::Mat const &left_mask,
                           cv::Mat const &right_mask,
                           int const blocksize)
{
    assert(blocksize > 0);

    assert(left.size == right.size);
    assert(right.size == left_mask.size);
    assert(left_mask.size == right_mask.size);

    auto const center_index = blocksize / 2;
    assert(center_index > 0);

    // The disparity is a measure of distance of a point to an arbitrary focus
    // plane. Therefore the disparity can be signed to reflect point behind
    // plane and point in front of plane cases. Therefore we need to divide the
    // maximum displacement interval into both cases.
    auto const displ_count = m_displacements_per_pixel;
    auto const displ_middle = displ_count / 2;
    auto const displ_begin = -(displ_count - displ_middle);

    cv::Mat const left_gray;
    cv::Mat const right_gray;
    cv::cvtColor(left, const_cast<cv::Mat &>(left_gray), cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, const_cast<cv::Mat &>(right_gray), cv::COLOR_BGR2GRAY);

    // in every scanline, every pixel should be checked for every displacement
    //#pragma omp parallel for collapse(2)
    for (int y = 0; y < left_gray.rows; y++) {
        for (int x = 0; x < left_gray.cols; x++) {
            // The first block against which the second will be tested
            cv::Rect const left_block =
                    clamp_into({x - center_index,
                                y - center_index,
                                blocksize,
                                blocksize},
                               left_gray);

            for (int d = displ_begin; d < displ_middle; d++) {
                // On edges where the change in disparity changes the clamped
                // shape of the ROI rectangle, it must be ensured, that the
                // blocks still match
                cv::Rect const right_block =
                        clamp_into({left_block.x - d,
                                    left_block.y,
                                    left_block.width,
                                    left_block.height},
                                   left_gray);
                cv::Rect block = overlap({left_block, right_block});

                auto const left_roi = left_gray(block);
                auto const right_roi = right_gray(block);
                // use sum of absolute difference as disparity metric
                cv::Mat const differences(
                            cv::Size(left_roi.cols, left_roi.rows),
                            left_roi.type(), // signed distance !!
                            cv::Scalar(0));
                cv::absdiff(left_roi, // macht das wirklich das richtige?
                            right_roi,
                            const_cast<cv::Mat &>(differences));

                // the difference image should only be gray and therefore the
                // SAD is in the first channel of the sum image
                assert(differences.channels() == 1);
                int const SAD =
                    static_cast<int>(std::round(cv::sum(differences)[0]));
                assert(SAD >= 0);

                // NOTE As all cost values were preallocated, assume the write
                // access to individual ints in the vector is threadsafe
                m_cost_volume.at(to_linear(y, x, d + std::abs(displ_begin)))
                    = SAD;
            }
        }
    }
}

size_t CostVolume::to_linear(int const scanline, int const x, int const d) const
{
    assert(scanline >= 0);
    assert(x >= 0);
    assert(d >= 0);
    assert(scanline < m_scanline_count);
    assert(x < m_pixels_per_scanline);
    assert(d < m_displacements_per_pixel);

    // TODO: Check for overflow, as qubic to linear conversion exhibits a high
    // probability for this UB

    // To get the linear array index sum up along
    // scanlines -> pixels -> disparity
    auto const index{
        d +
        x * m_displacements_per_pixel +
        scanline * m_pixels_per_scanline * m_displacements_per_pixel};

    assert(index >= 0);
    assert(static_cast<size_t>(index) < m_cost_volume.size());
    return static_cast<size_t>(index);
}

bool CostVolume::is_masked(const cv::Point2i &pixel) const
{
    assert(is_valid());

    assert(m_mask.type() == CV_8UC1);
    assert(pixel.x >= 0);
    assert(pixel.y >= 0);

    return m_mask.at<uint8_t>(pixel) > 0;
}

cv::Mat CostVolume::slice(const int scanline) const
{
    assert(is_valid());
    assert(scanline < slice_count());

    // Reject all invalid scanline indices
    if (scanline >= slice_count())
        return cv::Mat();

    auto const max_displacement = m_displacements_per_pixel;
    auto const width = m_pixels_per_scanline;

    cv::Mat slice(cv::Size(width, max_displacement),
                  CV_8UC1,
                  cv::Scalar(0));

    for (int displacement = 0; displacement < max_displacement; ++displacement)
        for (int x = 0; x < width; ++x) {
            const auto matching_cost =
                    m_cost_volume.at(
                        to_linear(
                            scanline,
                            x,
                            displacement
                            )
                        );

            slice.at<uint8_t>(displacement, x) =
                matching_cost > std::numeric_limits<uint8_t>::max()
                ? std::numeric_limits<uint8_t>::max()
                : static_cast<uint8_t>(matching_cost);
        }

    return slice;
}

int CostVolume::slice_count() const
{
    assert(is_valid());

    return m_scanline_count;
}

bool CostVolume::is_valid() const
{
    bool const has_volume = (m_scanline_count > 0)
                            && (m_pixels_per_scanline > 0)
                            && (m_displacements_per_pixel > 0);

    auto const volume {
        m_scanline_count *
        m_pixels_per_scanline *
        m_displacements_per_pixel
    };

    assert(volume >= 0);

    return (has_volume)
           && (static_cast<size_t>(volume) == m_cost_volume.size());
}
