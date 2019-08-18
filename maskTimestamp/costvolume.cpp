#include "costvolume.h"

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>

#include <opencv2/core/core.hpp>

CostVolume::CostVolume(cv::Rect const &size, size_t const d)
    : m_cost_volume(static_cast<size_t>(size.width * size.height) * d, 0)
    , m_max_displacement(d)
{
    assert(size.height > 0);
    assert(size.width > 0);
    assert(d > 0);
}

void CostVolume::calculate(const cv::Mat &left,
                           const cv::Mat &right,
                           const cv::Mat &mask,
                           const size_t blocksize)
{
    assert(blocksize % 2);
    assert(left.size == right.size);
    assert(right.size == mask.size);

    // If the blocksize is divisible by 2, the center pixel index in the roi is
    int const center_index = blocksize / 2;
    assert(center_index > 0);

    assert(std::numeric_limits<int>::max() >= m_max_displacement);
    auto const mxdsplcmnt = static_cast<int>(m_max_displacement);

    // in every scanline, every pixel should be checked for every displacement
    for (int y = 0; y < left.rows; y++)
        for (int x = 0; x < left.cols; x++)
            for (int d = 0; d < mxdsplcmnt; d++) {
                auto const blcksz = static_cast<int>(blocksize);

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
                m_cost_volume.at(to_linear(static_cast<size_t>(x),
                                           static_cast<size_t>(y),
                                           d,
                                           static_cast<size_t>(left.rows),
                                           static_cast<size_t>(left.cols)))
                    = SAD;
            }
}

size_t CostVolume::to_linear(size_t const x,
                             size_t const scanline,
                             size_t const d,
                             size_t const scanline_dim,
                             size_t const x_dim) const
{
    // To get the linear array index sum up along
    // scanlines -> pixels -> disparity
    return (scanline + x * scanline_dim + d * scanline_dim * x_dim);
}

cv::Mat CostVolume::slice(const size_t y) const
{
    assert(is_valid());

    assert(std::numeric_limits<int>::max >= m_max_displacement);
    auto const max_displacement = static_cast<int>(m_max_displacement);

    cv::Mat slice(max_displacement, m_size.width, CV_8S);

    for (int displacement = 0; displacement != max_displacement; displacement++)
        for (int x = 0; x != m_size.width; x++) {
            assert(displacement >= 0);
            assert(x >= 0);
            assert(m_size.width);

            const auto disparity_value = m_cost_volume.at(
                to_linear(static_cast<size_t>(x),
                          y,
                          static_cast<size_t>(displacement),
                          m_max_displacement,
                          static_cast<size_t>(m_size.width)));
            slice.at<int8_t>(displacement, x) = disparity_value; // TODO !!
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
                   == m_cost_volume.size());
}
