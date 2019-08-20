#include "costvolume.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <forward_list>
#include <numeric>
#include <type_traits>

#include <opencv2/core/core.hpp>
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
    cv::Point2i const first(
        std::clamp(r.x, region.x, region.x + region.width),
        std::clamp(r.y, region.y, region.y + region.height));

    cv::Point2i const second(
        std::clamp(r.x + r.width, region.x, region.x + region.width),
        std::clamp(r.y + r.height, region.y, region.y + region.height));

    assert(region.contains(first));
    assert(region.contains(second));

    return cv::Rect(first, second);
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
                  [region](cv::Rect r) { clamp_into(r, region); });

    return region;
}

CostVolume::CostVolume(cv::Rect const &size, size_t const d)
    : m_cost_volume(static_cast<size_t>(size.width * size.height) * d, 0)
    , m_max_displacement(d)
    , m_size(size)
{
    assert(size.height > 0);
    assert(size.width > 0);
    assert(d > 0);
}

void CostVolume::calculate(cv::Mat const &left,
                           cv::Mat const &right,
                           cv::Mat const &left_mask,
                           cv::Mat const &right_mask,
                           size_t const block_size)
{
    assert(block_size % 2);
    assert(left.size == right.size);
    assert(right.size == left_mask.size);
    assert(left_mask.size == right_mask.size);

    int const center_index = narrow<int8_t>(block_size / 2);
    assert(center_index > 0);

    auto const mxdsplcmnt = narrow<int>(m_max_displacement);

    cv::Mat const left_gray;
    cv::Mat const right_gray;
    cv::cvtColor(left, const_cast<cv::Mat &>(left_gray), cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, const_cast<cv::Mat &>(right_gray), cv::COLOR_BGR2GRAY);

    // in every scanline, every pixel should be checked for every displacement
    for (int y = 0; y < left_gray.rows; y++)
        for (int x = 0; x < left_gray.cols; x++)
            for (int d = 0; d < mxdsplcmnt; d++) {
                auto const blcksz = narrow<int>(block_size);

                cv::Rect const left_block =
                    clamp_into({x - center_index,
                                          y - center_index,
                                          blcksz,
                                blcksz}, left_gray);
                // On edges where the change in disparity changes the clamped
                // shape of the ROI rectangle, it must be ensured, that the
                // blocks still match
                cv::Rect const right_block(left_block.x - d,
                                           left_block.y - d,
                                           left_block.width,
                                           left_block.height);

                assert(right_block == clamp_into(right_block, right_gray));

                auto const left_roi =
                    left_gray(clamp_into(left_block, left_gray));
                auto const right_roi =
                    right_gray(clamp_into(right_block, right_gray));

                // use sum of absolute difference as disparity metric
                cv::Mat const differences;
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
                       !std::signbit(d) &&
                       !std::signbit(left_gray.rows) &&
                       !std::signbit(left_gray.cols));
                m_cost_volume.at(to_linear(static_cast<size_t>(y),
                                           static_cast<size_t>(x),
                                           d,
                                           static_cast<size_t>(left_gray.rows),
                                           static_cast<size_t>(left_gray.cols)))
                    = SAD;
            }
}

size_t CostVolume::to_linear(size_t const scanline,
                             size_t const x,
                             size_t const d,
                             size_t const scanline_dim,
                             size_t const x_dim) const
{
    // To get the linear array index sum up along
    // scanlines -> pixels -> disparity
    return (scanline + x * scanline_dim + d * scanline_dim * x_dim);
}

bool CostVolume::is_masked(const cv::Point2i &pixel) const
{
    assert(is_valid());

    assert(m_mask.type() == CV_8UC1);
    assert(pixel.x >= 0);
    assert(pixel.y >= 0);

    return m_mask.at<uint8_t>(pixel) > 0;
}

cv::Mat CostVolume::slice(const size_t y) const
{
    assert(is_valid());

    // Reject all invalid scanline indices
    if (y >= slice_count())
        return cv::Mat();

    auto const max_displacement = narrow<int>(m_max_displacement);

    cv::Mat slice(max_displacement, m_size.width, CV_8SC1);

    for (int displacement = 0;
         displacement != 2 * max_displacement + 1;
         ++displacement)
        for (int x = 0; x != m_size.width; ++x) {
            const auto matching_cost = m_cost_volume.at(
                to_linear(y,
                          static_cast<size_t>(x),
                          static_cast<size_t>(displacement),
                          m_max_displacement,
                          static_cast<size_t>(m_size.width)));
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

    assert(m_size.height >= 0);
    return static_cast<size_t>(m_size.height);
}

bool CostVolume::is_valid() const
{
    assert(m_size.x >= 0);
    assert(m_size.y >= 0);
    return (m_size.area() > 0)
           && (static_cast<size_t>(m_size.x)
                   * static_cast<size_t>(m_size.y)
                   * m_max_displacement
                   == m_cost_volume.size())
           && (m_size == cv::Rect());
}
