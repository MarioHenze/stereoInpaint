#include "costvolume.h"

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include <opencv2/core/core.hpp>

template<typename U, typename T>
U narrow(T const big)
{
    static_assert(std::is_integral<T>::value, "Type T needs to be integral");
    static_assert(std::is_unsigned<T>::value, "Type T needs to be unsigned");
    static_assert(std::is_integral<U>::value, "Type U needs to be integral");

    assert(std::numeric_limits<U>::max() >= big);
    return static_cast<U>(big);
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

    // in every scanline, every pixel should be checked for every displacement
    for (int y = 0; y < left.rows; y++)
        for (int x = 0; x < left.cols; x++)
            for (int d = 0; d < mxdsplcmnt; d++) {
                auto const blcksz = narrow<int>(block_size);

                auto const left_roi = left({x - center_index,
                                            y - center_index,
                                            blcksz,
                                            blcksz});
                auto const right_roi = right({x - center_index - d,
                                              y - center_index - d,
                                              blcksz,
                                              blcksz});

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
                assert(!std::signbit(x) && !std::signbit(y) && !std::signbit(d)
                       && !std::signbit(left.rows) && !std::signbit(left.cols));
                m_cost_volume.at(to_linear(static_cast<size_t>(y),
                                           static_cast<size_t>(x),
                                           d,
                                           static_cast<size_t>(left.rows),
                                           static_cast<size_t>(left.cols)))
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
    assert(m_size.height >= 0);
    auto const volume_height = static_cast<size_t>(m_size.height);
    if (y >= volume_height)
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
