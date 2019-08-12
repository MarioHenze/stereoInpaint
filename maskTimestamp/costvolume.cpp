#include "costvolume.h"

CostVolume::CostVolume(const cv::Rect &size, const size_t d):
    m_cost_volume(static_cast<size_t>(size.width * size.height) * d, 0),
    m_max_displacement(d)
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

    // in every scanline, every pixel should be checked for every displacement
    for (int y = 0; y < left.rows; y++)
    for (int x = 0; x < left.cols; x++)
    for (size_t d = 0; d < m_max_displacement; d++) {

        auto const left_roi = left({x - center_index,
                                    y - center_index,
                                    static_cast<int>(blocksize),
                                    static_cast<int>(blocksize)});
        auto const right_roi = right({x - center_index - static_cast<int>(d),
                                      y - center_index - static_cast<int>(d),
                                      static_cast<int>(blocksize),
                                      static_cast<int>(blocksize)});
    }

}
