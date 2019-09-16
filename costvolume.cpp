#include "costvolume.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <forward_list>
#include <limits>
#include <list>
#include <numeric>
#include <type_traits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

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

cv::Size overlap(std::forward_list<cv::Rect> rects)
{
    if (rects.empty())
        return cv::Size();

    cv::Size common_minimum = rects.front().size();

    for (auto const &r : rects) {
        common_minimum.width = std::min(common_minimum.width, r.width);
        common_minimum.height = std::min(common_minimum.height, r.height);
    }

    return common_minimum;
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

    m_mask_left = left_mask.clone();
    m_mask_right = right_mask.clone();

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
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < left_gray.rows; y++) {
        for (int x = 0; x < left_gray.cols; x++) {
            // The first block against which the second will be tested
            cv::Rect left_block = clamp_into({x - center_index,
                                              y - center_index,
                                              blocksize,
                                              blocksize},
                                             left_gray);

            for (int d = displ_begin; d < displ_middle; d++) {
                // On edges where the change in disparity changes the clamped
                // shape of the ROI rectangle, it must be ensured, that the
                // blocks still match
                cv::Rect right_block = clamp_into({left_block.x - d,
                                                   left_block.y,
                                                   left_block.width,
                                                   left_block.height},
                                                  left_gray);
                auto common_size = overlap({left_block, right_block});

                // Therefore modify the size of the ROIs
                left_block.width = common_size.width;
                left_block.height = common_size.height;
                right_block.width = common_size.width;
                right_block.height = common_size.height;

                auto const left_roi = left_gray(left_block);
                auto const right_roi = right_gray(right_block);
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

    // The cost volume is now calculated, but at the partially masked locations
    // we need to linearily interpolate the disparity value
//    for (int y = 0; y < left_mask.rows; ++y) {
//        for (int x = 0; x < left_mask.cols; ++x) {
//            // If both or none images are masked, there is nothing to do
//            auto const left_mask_pixel = left_mask.at<uint8_t>(x, y);
//            auto const right_mask_pixel = right_mask.at<uint8_t>(x, y);
//            if ((left_mask_pixel > 0 && right_mask_pixel > 0)
//                || (left_mask_pixel == 0 && right_mask_pixel == 0))
//                continue;

//            // As we can only go from no <-> partial <-> full occlusion, we know
//            // the last unoccluded pixel was one before this loop
//            auto last_left_disparity = x - 1;
//        }
//    }
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

    //assert(m_mask.type() == CV_8UC1);
    assert(pixel.x >= 0);
    assert(pixel.y >= 0);

    return false;//m_mask.at<uint8_t>(pixel) > 0;
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
                  CV_16UC1,
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

            slice.at<uint16_t>(displacement, x) =
                matching_cost > std::numeric_limits<uint16_t>::max()
                ? std::numeric_limits<uint16_t>::max()
                : static_cast<uint16_t>(matching_cost);
        }

    return slice;
}

int CostVolume::slice_count() const
{
    assert(is_valid());

    return m_scanline_count;
}

std::vector<int> CostVolume::trace_disparity(const cv::Mat cost_slice) const
{
    assert(cost_slice.cols > 0);
    assert(cost_slice.rows > 0);

    // The accumulated cost map for the dynamic programming
    cv::Mat D = cost_slice.clone();
    D.convertTo(D, CV_32SC1);

    for (int i = 1; i < D.cols; ++i) {
        auto const prev_accumulated_col = D.col(i - 1);
        auto const match_cost_col = D.col(i);

        for (int d = 0; d < cost_slice.rows; ++d) {
            // We need a vector which represents the cost of changing the
            // disparity value
            std::vector<int> change_cost(static_cast<size_t>(cost_slice.rows),
                                         0);
            std::iota(change_cost.begin(), change_cost.end(), -d);
            // Our distance metric for changing the disparity value is x^2
            std::for_each(change_cost.begin(),
                          change_cost.end(),
                          [](int const x) -> int { return std::pow(x, 2); });
            // The cost is composed of the block matching cost, the accumulated
            // cost from a previous cell and the step cost
            auto sum = prev_accumulated_col.clone();
            auto const cost_mat = cv::Mat(change_cost.size(),
                                          1,
                                          CV_32SC1,
                                          change_cost.data());
            cv::add(prev_accumulated_col,
                    cost_mat,
                    sum);

            double min{0};
            cv::minMaxIdx(sum, &min);
            assert(static_cast<double>(std::numeric_limits<int32_t>::max())
                   > min);
            cv::add(match_cost_col, min, D.col(i));
        }
    }

    std::vector<int> disparity(static_cast<size_t>(cost_slice.cols), 0);
    // Now D contains the accumulated cost and the disparity corresponds with
    // the y positions of the minimal path from right to left
    for (int i = D.cols - 1; i >= 0; --i) {
        int d{0};
        cv::minMaxIdx(D.col(i), nullptr, nullptr, &d);
        assert(d >= 0);
        disparity.at(static_cast<size_t>(i)) = d;
    }

    return disparity;
}

cv::Mat CostVolume::calculate_disparity_map() const
{
    assert(m_mask_left.type() == CV_8UC1);

    cv::Mat disparity_map(m_mask_left.rows, m_mask_left.cols, CV_32S);

    // In every scanline we need to calculate the disparity line in the known
    // regions and interpolate over masked intervals along the known endpoints
    for (int scanline = 0; scanline < m_scanline_count; ++scanline) {
        auto const cost_slice = slice(scanline);

        std::vector<std::pair<int, int>> intervals;
        int valid_count{0};
        for (int pixel = 0; pixel < m_pixels_per_scanline; ++pixel) {
            // is masked?
            auto const pixel_value = m_mask_left.at<uint8_t>(scanline, pixel);
            if (pixel_value == 0) {
                valid_count++;
            } else {
                // We've encountered a masked pixel and valid_count gives us the
                // witdh of the interval containing unmasked pixels
                intervals.emplace_back(pixel - valid_count, pixel);
                valid_count = 0;
            }
        }

        // If the scanline ends on a valid pixel insert interval
        if (valid_count > 0) {
            intervals.emplace_back(m_pixels_per_scanline - valid_count,
                                   m_pixels_per_scanline);
        }

        auto const same_pred = [](std::pair<int, int> const interval) -> bool {
            return interval.second == interval.first;
        };
        // All entries where second == first result from masked cols
        intervals.erase(std::remove_if(intervals.begin(),
                                       intervals.end(),
                                       same_pred),
                        intervals.end());

        // Contains all the disparity traces from the known regions
        std::vector<std::vector<int>> traces;

        for (auto const &interval : intervals) {
            // This mat contains the continous intervals of the cost slice,
            // which are not masked
            auto const sub_slice = cost_slice.colRange(interval.first,
                                                       interval.second);
            traces.push_back(trace_disparity(sub_slice));
        }

        // clamp disparity value on front of scaline
        if (intervals.front().first > 0) {
            auto const value = traces.front().front();
            disparity_map.row(scanline).colRange(0, intervals.front().first)
                = value;
        }

        assert(traces.size() == intervals.size());
        auto const lerp = [](int const a, int const b, float const t) -> int {
            auto const result = std::lround(a + t * (b - a));
            assert(std::numeric_limits<int>::max() >= result);
            return static_cast<int>(result);
        };

        while (!intervals.empty()) {
            auto const interval = intervals.front();
            // The calculated disparity trace needs to be inserted
            auto const trace_mat = cv::Mat(1, traces.front().size(), CV_32S, traces.front().data());
//            disparity_map.row(scanline)
//                .colRange(interval.first, interval.second)
//                .setTo(trace_mat);
            trace_mat.copyTo(
                disparity_map.row(scanline).colRange(interval.first,
                                                     interval.second));
            //                .setTo(trace_mat))
            // Interpolate between the traces
            if ((traces.size() > 1) && (intervals.size() > 1)) {
                auto const last_prev_value = traces.front().back();
                auto const first_next_value = traces.at(1).front();
                auto row = disparity_map
                               .row(scanline)
                               .colRange(interval.second,
                                         intervals.at(1).first);
                for (int i = 0; i < row.cols; ++i) {
                    row.at<int32_t>(i) = lerp(last_prev_value,
                                              first_next_value,
                                              i / static_cast<float>(row.cols));
                }
            }
            intervals.erase(intervals.begin());
            traces.erase(traces.begin());
        }
    }

    // TODO add offset !!

    return disparity_map;
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
