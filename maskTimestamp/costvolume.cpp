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
        std::clamp(r.x, region.x, region.x + region.width - 1),
        std::clamp(r.y, region.y, region.y + region.height - 1));

    cv::Point2i const lower_right(
        std::clamp(r.x + r.width - 1, region.x, region.x + region.width - 1),
        std::clamp(r.y + r.height - 1, region.y, region.y + region.height - 1));

    assert(region.contains(upper_left));
    assert(region.contains(lower_right));

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

CostVolume::CostVolume(size_t const width, size_t const height, size_t const d)
    : m_cost_volume(width * height * (2 * d + 1), 0)
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
                           cv::Mat const &right_mask)
{
    assert(left.size == right.size);
    assert(right.size == left_mask.size);
    assert(left_mask.size == right_mask.size);

    int const center_index = narrow<int8_t>(m_displacements_per_pixel / 2);
    assert(center_index > 0);

    auto const mxdsplcmnt = narrow<int>(m_displacements_per_pixel);

    cv::Mat const left_gray;
    cv::Mat const right_gray;
    cv::cvtColor(left, const_cast<cv::Mat &>(left_gray), cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, const_cast<cv::Mat &>(right_gray), cv::COLOR_BGR2GRAY);

    // in every scanline, every pixel should be checked for every displacement
    for (int y = 0; y < left_gray.rows; y++)
        for (int x = 0; x < left_gray.cols; x++)
            #pragma omp parallel for
            for (int d = 0; d < mxdsplcmnt; d++) {
                cv::Rect const left_block =
                    clamp_into({x - center_index,
                                y - center_index,
                                mxdsplcmnt,
                                mxdsplcmnt}, left_gray);
                // On edges where the change in disparity changes the clamped
                // shape of the ROI rectangle, it must be ensured, that the
                // blocks still match
                cv::Rect const right_block(left_block.x - d,
                                           left_block.y - d,
                                           left_block.width,
                                           left_block.height);
                cv::Rect block = overlap({left_block, right_block});

                auto const left_roi = left_gray(block);
                auto const right_roi = right_gray(block);

                // use sum of absolute difference as disparity metric
                cv::Mat const differences(
                            cv::Size(left_roi.cols, left_roi.rows),
                            left_roi.type(),
                            cv::Scalar(0));
                cv::absdiff(left_roi,
                            right_roi,
                            const_cast<cv::Mat &>(differences));

                // the difference image should only be gray and therefore the
                // SAD is in the first channel of the sum image
                assert(differences.channels() == 1);
                size_t const SAD =
                    static_cast<size_t>(std::round(cv::sum(differences)[0]));

                // if all coordinates are valid (i.e. positive) a conversation
                // to an unsigned integral type can be made
                assert(!std::signbit(x) &&
                       !std::signbit(y) &&
                       !std::signbit(d));

                // NOTE As all cost values were preallocated, assume the write
                // access to individual ints in the vector is threadsafe
                m_cost_volume.at(
                            to_linear(static_cast<size_t>(y),
                                      static_cast<size_t>(x),
                                      d))
                        = SAD;
            }
}

size_t CostVolume::to_linear(size_t const scanline,
                             size_t const x,
                             size_t const d) const
{
    assert(scanline < m_scanline_count);
    assert(x < m_pixels_per_scanline);
    assert(d < m_displacements_per_pixel);

    // To get the linear array index sum up along
    // scanlines -> pixels -> disparity
    const auto index{
        d +
        x * m_displacements_per_pixel +
        scanline * m_pixels_per_scanline * m_displacements_per_pixel};

    assert(index < m_cost_volume.size());
    return index;
}

bool CostVolume::is_masked(const cv::Point2i &pixel) const
{
    assert(is_valid());

    assert(m_mask.type() == CV_8UC1);
    assert(pixel.x >= 0);
    assert(pixel.y >= 0);

    return m_mask.at<uint8_t>(pixel) > 0;
}

cv::Mat CostVolume::slice(const size_t scanline) const
{
    assert(is_valid());

    // Reject all invalid scanline indices
    if (scanline >= slice_count())
        return cv::Mat();

    auto const max_displacement = narrow<int>(m_displacements_per_pixel);
    auto const width = narrow<int>(m_pixels_per_scanline);

    cv::Mat slice(cv::Size(width, max_displacement),
                  CV_8SC1,
                  cv::Scalar(0));

    for (int displacement = 0; displacement < max_displacement; ++displacement)
        for (int x = 0; x < width; ++x) {
            const auto matching_cost =
                    m_cost_volume.at(
                        to_linear(
                            scanline,
                            static_cast<size_t>(x),
                            static_cast<size_t>(displacement)
                            )
                        );

            slice.at<uint8_t>(displacement, x) =
                matching_cost > std::numeric_limits<uint8_t>::max()
                ? std::numeric_limits<uint8_t>::max()
                : static_cast<uint8_t>(matching_cost);
        }

    return slice;
}

size_t CostVolume::slice_count() const
{
    assert(is_valid());

    return m_scanline_count;
}

bool CostVolume::is_valid() const
{
    auto const volume {
        m_scanline_count *
        m_pixels_per_scanline *
        m_displacements_per_pixel
    };

    return (volume > 0) && (volume == m_cost_volume.size());
}
