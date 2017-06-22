#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

template <typename intT>
double d_chebyshev(const std::vector<intT>& histogram)
{
  double NumBins   = 0;
  double NumEvents = 0;
  for (unsigned i=0; i<histogram.size(); i++) {
    if (histogram.at(i)>0) {
      NumEvents += histogram.at(i);
      NumBins   += 1.0;
    }
  }
  double expectation = NumEvents/NumBins;
  double d_cheby = 0;
  for (unsigned i=0; i<histogram.size(); i++) {
    if (histogram.at(i)>0) {
      double deviation = expectation - histogram.at(i);
      if (fabs(deviation)>d_cheby)
        d_cheby=fabs(deviation);
    }
  }
  return d_cheby/expectation;
}

template <typename intT>
double d_kullback(const std::vector<intT>& histogram)
{
  double NumBins   = 0;
  double NumEvents = 0;
  for (unsigned i=0; i<histogram.size(); i++) {
    if (histogram.at(i)>0) {
      NumEvents += histogram.at(i);
      NumBins   += 1.0;
    }
  }
  double d_kullback = 0;
  for (unsigned i=0; i<histogram.size(); i++) {
    if (histogram.at(i)>0) {
      double P_i=histogram.at(i)/NumEvents;
      double Q_i=1/NumBins;
      d_kullback += P_i*log(P_i/Q_i);
    }
  }
  return d_kullback;
}

// determine the first and last index with non-empty histogram entry
template <typename intT>
void getHistogramRange(const std::vector<intT>& histogram, size_t& start, size_t& end)
{
  start = histogram.size();
  end = 0;
  for (size_t i=0; i<histogram.size(); i++) {
    if (histogram.at(i) > 1) {
      if (i < start){
        start = i;
      }
      break;
    }
  }
  for (size_t i=histogram.size(); i-- > 0; ) {
    if (histogram.at(i) > 1) {
      if (i > end){
        end = i;
      }
      break;
    }
  }
}

// update muca weights according to W^{n+1}(E) = W^{n}(E) / H^{n}(E)
template <typename intT>
void updateWeights(std::vector<float>& weights, const std::vector<intT>& histogram)
{
  size_t hist_start, hist_end;
  getHistogramRange(histogram, hist_start, hist_end);
  for (size_t i=hist_start; i<=hist_end; ++i) {
    if (histogram.at(i) > 0){
      weights.at(i) = weights.at(i) - log((float)(histogram.at(i)));
    }
  }
}

// custom modification of weights after update (use with care)
template <typename intT>
bool modifyWeights(std::vector<float>& log_weights, const std::vector<intT>& histogram, const size_t slope_range, const int shift_index=-1)
{
  size_t hist_start, hist_end;
  getHistogramRange(histogram, hist_start, hist_end);
  if (hist_start == histogram.size() || hist_end == 0) {
    return false;
  }

  // linear extrapolation from hist_start to lower bound of histogram
  if (hist_start + slope_range < histogram.size()) {
    float slope = (log_weights.at(hist_start)-log_weights.at(hist_start+slope_range))/slope_range;
    for (size_t i=0; i<hist_start; ++i) {
      log_weights.at(i) = log_weights.at(hist_start)+slope*(hist_start-i);
    }
  }

  // linear extrapolation from hist_end to upper bound of histogram
  if (hist_end > slope_range) {
    float slope = (log_weights.at(hist_end)-log_weights.at(hist_end-slope_range))/slope_range;
    for (size_t i=hist_end; i<log_weights.size(); ++i) {
      log_weights.at(i) = log_weights.at(hist_end)+slope*(i-hist_end);
    }
  }

  // constant shift setting log_weight(shift_index) to zero
  if (shift_index != -1) {
    float shift = log_weights.at(shift_index);
    for (size_t i=0; i<log_weights.size(); ++i) {
      log_weights.at(i) -= shift;
    }
  }

  return true;
}
