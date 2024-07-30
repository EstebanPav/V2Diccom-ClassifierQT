/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <fstream>

#include "random_forest/utility.h"
#include "TreeSurvival.h"
#include "random_forest/Data.h"

namespace ranger {

TreeSurvival::TreeSurvival(double* unique_timepoints, size_t* response_timepointIDs, size_t num_timepoints) :
    unique_timepoints(unique_timepoints), response_timepointIDs(response_timepointIDs), num_deaths(nullptr), num_samples_at_risk(nullptr), num_timepoints(num_timepoints) {
}

TreeSurvival::TreeSurvival(size_t** child_nodeIDs, size_t* split_varIDs, double* split_values, double** chf,
    double* unique_timepoints, size_t* response_timepointIDs, size_t num_timepoints) :
    Tree(child_nodeIDs, split_varIDs, split_values), unique_timepoints(unique_timepoints), response_timepointIDs(response_timepointIDs), chf(chf), num_deaths(nullptr), num_samples_at_risk(nullptr), num_timepoints(num_timepoints) {
}

void TreeSurvival::allocateMemory() {
    // Number of deaths and samples at risk for each timepoint
    num_deaths = new size_t[num_timepoints];
    num_samples_at_risk = new size_t[num_timepoints];
}

void TreeSurvival::appendToFileInternal(std::ofstream& file) {  // #nocov start

    // Convert to dynamic arrays without empty elements and save
    size_t* terminal_nodes = new size_t[num_timepoints];
    size_t terminal_nodes_count = 0;
    double** chf_array = new double*[num_timepoints];
    size_t chf_array_count = 0;

    for (size_t i = 0; i < num_timepoints; ++i) {
        if (chf[i] != nullptr) {
            terminal_nodes[terminal_nodes_count++] = i;
            chf_array[chf_array_count] = new double[num_timepoints];
            std::copy(chf[i], chf[i] + num_timepoints, chf_array[chf_array_count]);
            ++chf_array_count;
        }
    }

    saveArray1D(terminal_nodes, terminal_nodes_count, file);
    saveArray2D(chf_array, chf_array_count, num_timepoints, file);

    // Clean up
    delete[] terminal_nodes;
    for (size_t i = 0; i < chf_array_count; ++i) {
        delete[] chf_array[i];
    }
    delete[] chf_array;

} // #nocov end

void TreeSurvival::createEmptyNodeInternal() {
    chf = new double*[num_timepoints];
    for (size_t i = 0; i < num_timepoints; ++i) {
        chf[i] = nullptr;
    }
}

void TreeSurvival::computeSurvival(size_t nodeID) {
    double* chf_temp = new double[num_timepoints];
    double chf_value = 0;
    for (size_t i = 0; i < num_timepoints; ++i) {
        if (num_samples_at_risk[i] != 0) {
            chf_value += static_cast<double>(num_deaths[i]) / static_cast<double>(num_samples_at_risk[i]);
        }
        chf_temp[i] = chf_value;
    }
    chf[nodeID] = chf_temp;
}

double TreeSurvival::computePredictionAccuracyInternal(double* prediction_error_casewise) {

    // Compute summed chf for samples
    double* sum_chf = new double[prediction_terminal_nodeIDs.size()];
    size_t sum_chf_count = 0;
    for (size_t i = 0; i < prediction_terminal_nodeIDs.size(); ++i) {
        size_t terminal_nodeID = prediction_terminal_nodeIDs[i];
        double sum = std::accumulate(chf[terminal_nodeID], chf[terminal_nodeID] + num_timepoints, 0.0);
        sum_chf[sum_chf_count++] = sum;
    }

    // Return concordance index
    double result = computeConcordanceIndex(*data, sum_chf, oob_sampleIDs, prediction_error_casewise);

    // Clean up
    delete[] sum_chf;

    return result;
}

bool TreeSurvival::splitNodeInternal(size_t nodeID, size_t* possible_split_varIDs, size_t num_possible_split_varIDs) {

    // Stop if node is pure
    bool pure = true;
    double pure_time = 0;
    double pure_status = 0;
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        double time = data->get_y(sampleID, 0);
        double status = data->get_y(sampleID, 1);
        if (pos != start_pos[nodeID] && (time != pure_time || status != pure_status)) {
            pure = false;
            break;
        }
        pure_time = time;
        pure_status = status;
    }
    if (pure) {
        computeDeathCounts(nodeID);
        computeSurvival(nodeID);
        return true;
    }

    if (splitrule == MAXSTAT) {
        return findBestSplitMaxstat(nodeID, possible_split_varIDs, num_possible_split_varIDs);
    } else if (splitrule == EXTRATREES) {
        return findBestSplitExtraTrees(nodeID, possible_split_varIDs, num_possible_split_varIDs);
    } else {
        return findBestSplit(nodeID, possible_split_varIDs, num_possible_split_varIDs);
    }
}

}

/////////////////////////////////////////////////////////////////////////////
bool TreeSurvival::findBestSplit(size_t nodeID, size_t* possible_split_varIDs, size_t num_split_vars) {

  double best_decrease = -1;
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  size_t best_varID = 0;
  double best_value = 0;

  computeDeathCounts(nodeID);

  // Stop if maximum node size or depth reached (will check again for each child node)
  if (num_samples_node <= min_node_size || (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth)) {
    computeSurvival(nodeID);
    return true;
  }

  // Stop early if no split possible
  if (num_samples_node >= 2 * min_node_size) {

    // For all possible split variables
    for (size_t i = 0; i < num_split_vars; ++i) {
      size_t varID = possible_split_varIDs[i];

      // Find best split value, if ordered consider all values as split values, else all 2-partitions
      if (data->isOrderedVariable(varID)) {
        if (splitrule == LOGRANK) {
          findBestSplitValueLogRank(nodeID, varID, best_value, best_varID, best_decrease);
        } else if (splitrule == AUC || splitrule == AUC_IGNORE_TIES) {
          findBestSplitValueAUC(nodeID, varID, best_value, best_varID, best_decrease);
        }
      } else {
        findBestSplitValueLogRankUnordered(nodeID, varID, best_value, best_varID, best_decrease);
      }

    }
  }

  // Stop and save CHF if no good split found (this is terminal node).
  if (best_decrease < 0) {
    computeSurvival(nodeID);
    return true;
  } else {
    // If not terminal node save best values
    split_varIDs[nodeID] = best_varID;
    split_values[nodeID] = best_value;

    // Compute decrease of impurity for this node and add to variable importance if needed
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
      addImpurityImportance(nodeID, best_varID, best_decrease);
    }

    // Regularization
    saveSplitVarID(best_varID);

    return false;
  }
}

bool TreeSurvival::findBestSplitMaxstat(size_t nodeID, size_t* possible_split_varIDs, size_t num_split_vars) {

  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];

  // Stop if maximum node size or depth reached
  if (num_samples_node <= min_node_size || (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth)) {
    computeDeathCounts(nodeID);
    computeSurvival(nodeID);
    return true;
  }

  // Compute scores
  double* time = new double[num_samples_node];
  double* status = new double[num_samples_node];
  size_t pos = 0;
  for (size_t i = start_pos[nodeID]; i < end_pos[nodeID]; ++i) {
    size_t sampleID = sampleIDs[i];
    time[pos] = data->get_y(sampleID, 0);
    status[pos] = data->get_y(sampleID, 1);
    ++pos;
  }
  std::vector<double> scores = logrankScores(std::vector<double>(time, time + num_samples_node), std::vector<double>(status, status + num_samples_node));

  // Save split stats
  double* pvalues = new double[num_split_vars];
  double* values = new double[num_split_vars];
  double* candidate_varIDs = new double[num_split_vars];
  double* test_statistics = new double[num_split_vars];

  size_t index = 0;

  // Compute p-values
  for (size_t i = 0; i < num_split_vars; ++i) {
    size_t varID = possible_split_varIDs[i];

    // Get all observations
    double* x = new double[num_samples_node];
    for (size_t j = start_pos[nodeID]; j < end_pos[nodeID]; ++j) {
      size_t sampleID = sampleIDs[j];
      x[j - start_pos[nodeID]] = data->get_x(sampleID, varID);
    }

    // Order by x
    std::vector<size_t> indices = order(std::vector<double>(x, x + num_samples_node), false);

    // Compute maximally selected rank statistics
    double best_maxstat;
    double best_split_value;
    maxstat(std::vector<double>(scores.begin(), scores.end()), std::vector<double>(x, x + num_samples_node), indices, best_maxstat, best_split_value, minprop, 1 - minprop);

    if (best_maxstat > -1) {
      // Compute number of samples left of cutpoints
      std::vector<size_t> num_samples_left = numSamplesLeftOfCutpoint(std::vector<double>(x, x + num_samples_node), indices);

      // Remove largest cutpoint (all observations left)
      num_samples_left.pop_back();

      // Use unadjusted p-value if only 1 split point
      double pvalue;
      if (num_samples_left.size() == 1) {
        pvalue = maxstatPValueUnadjusted(best_maxstat);
      } else {
        // Compute p-values
        double pvalue_lau92 = maxstatPValueLau92(best_maxstat, minprop, 1 - minprop);
        double pvalue_lau94 = maxstatPValueLau94(best_maxstat, minprop, 1 - minprop, num_samples_node,
            num_samples_left);

        // Use minimum of Lau92 and Lau94
        pvalue = std::min(pvalue_lau92, pvalue_lau94);
      }

      // Save split stats
      pvalues[index] = pvalue;
      values[index] = best_split_value;
      candidate_varIDs[index] = varID;
      test_statistics[index] = best_maxstat;
      ++index;
    }

    delete[] x;
  }

  double adjusted_best_pvalue = std::numeric_limits<double>::max();
  size_t best_varID = 0;
  double best_value = 0;
  double best_maxstat = 0;

  if (index > 0) {
    // Adjust p-values with Benjamini/Hochberg
    std::vector<double> adjusted_pvalues = adjustPvalues(std::vector<double>(pvalues, pvalues + index));

    double min_pvalue = std::numeric_limits<double>::max();
    for (size_t i = 0; i < index; ++i) {
      if (pvalues[i] < min_pvalue) {
        min_pvalue = pvalues[i];
        best_varID = static_cast<size_t>(candidate_varIDs[i]);
        best_value = values[i];
        adjusted_best_pvalue = adjusted_pvalues[i];
        best_maxstat = test_statistics[i];
      }
    }
  }

  delete[] time;
  delete[] status;
  delete[] pvalues;
  delete[] values;
  delete[] candidate_varIDs;
  delete[] test_statistics;

  // Stop and save CHF if no good split found (this is terminal node).
  if (adjusted_best_pvalue > alpha) {
    computeDeathCounts(nodeID);
    computeSurvival(nodeID);
    return true;
  } else {
    // If not terminal node save best values
    split_varIDs[nodeID] = best_varID;
    split_values[nodeID] = best_value;

    // Compute decrease of impurity for this node and add to variable importance if needed
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
      addImpurityImportance(nodeID, best_varID, best_maxstat);
    }

    return false;
  }
}

void TreeSurvival::computeDeathCounts(size_t nodeID) {

  // Initialize
  for (size_t i = 0; i < num_timepoints; ++i) {
    num_deaths[i] = 0;
    num_samples_at_risk[i] = 0;
  }

  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    double survival_time = data->get_y(sampleID, 0);

    size_t t = 0;
    while (t < num_timepoints && (*unique_timepoints)[t] < survival_time) {
      ++num_samples_at_risk[t];
      ++t;
    }

    // Now t is the survival time, add to at risk and to death if death
    if (t < num_timepoints) {
      ++num_samples_at_risk[t];
      if (data->get_y(sampleID, 1) == 1) {
        ++num_deaths[t];
      }
    }
  }
}


/////////////////////////////////////////////////////////////////////////////
void TreeSurvival::computeChildDeathCounts(size_t nodeID, size_t varID, double* possible_split_values,
    size_t* num_samples_right_child, size_t* delta_samples_at_risk_right_child,
    size_t* num_deaths_right_child, size_t num_splits) {

  // Count deaths in right child per timepoint and possible split
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    double value = data->get_x(sampleID, varID);
    size_t survival_timeID = (*response_timepointIDs)[sampleID];

    // Count deaths until split_value reached
    for (size_t i = 0; i < num_splits; ++i) {
      if (value > possible_split_values[i]) {
        ++num_samples_right_child[i];
        ++delta_samples_at_risk_right_child[i * num_timepoints + survival_timeID];
        if (data->get_y(sampleID, 1) == 1) {
          ++num_deaths_right_child[i * num_timepoints + survival_timeID];
        }
      } else {
        break;
      }
    }
  }
}

void TreeSurvival::findBestSplitValueLogRank(size_t nodeID, size_t varID, double& best_value, size_t& best_varID,
    double& best_logrank) {

  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];

  // Create possible split values
  double* possible_split_values = new double[num_samples_node];
  size_t num_split_values = data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Try next variable if all equal for this
  if (num_split_values < 2) {
    delete[] possible_split_values;
    return;
  }

  // -1 because no split possible at largest value
  size_t num_splits = num_split_values - 1;

  // Initialize
  size_t* num_deaths_right_child = new size_t[num_splits * num_timepoints]();
  size_t* delta_samples_at_risk_right_child = new size_t[num_splits * num_timepoints]();
  size_t* num_samples_right_child = new size_t[num_splits]();

  computeChildDeathCounts(nodeID, varID, possible_split_values, num_samples_right_child,
      delta_samples_at_risk_right_child, num_deaths_right_child, num_splits);

  // Compute logrank test for all splits and use best
  for (size_t i = 0; i < num_splits; ++i) {
    double numerator = 0;
    double denominator_squared = 0;

    // Stop if minimal node size reached
    size_t num_samples_left_child = num_samples_node - num_samples_right_child[i];
    if (num_samples_right_child[i] < min_node_size || num_samples_left_child < min_node_size) {
      continue;
    }

    // Compute logrank test statistic for this split
    size_t num_samples_at_risk_right_child = num_samples_right_child[i];
    for (size_t t = 0; t < num_timepoints; ++t) {
      if (num_samples_at_risk[t] < 2 || num_samples_at_risk_right_child < 1) {
        break;
      }

      if (num_deaths[t] > 0) {
        // Numerator and denominator for log-rank test, notation from Ishwaran et al.
        double di = static_cast<double>(num_deaths[t]);
        double di1 = static_cast<double>(num_deaths_right_child[i * num_timepoints + t]);
        double Yi = static_cast<double>(num_samples_at_risk[t]);
        double Yi1 = static_cast<double>(num_samples_at_risk_right_child);
        numerator += di1 - Yi1 * (di / Yi);
        denominator_squared += (Yi1 / Yi) * (1.0 - Yi1 / Yi) * ((Yi - di) / (Yi - 1)) * di;
      }

      // Reduce number of samples at risk for next timepoint
      num_samples_at_risk_right_child -= delta_samples_at_risk_right_child[i * num_timepoints + t];
    }
    double logrank = -1;
    if (denominator_squared != 0) {
      logrank = fabs(numerator / sqrt(denominator_squared));
    }

    // Regularization
    regularize(logrank, varID);

    if (logrank > best_logrank) {
      best_value = (possible_split_values[i] + possible_split_values[i + 1]) / 2;
      best_varID = varID;
      best_logrank = logrank;

      // Use smaller value if average is numerically the same as the larger value
      if (best_value == possible_split_values[i + 1]) {
        best_value = possible_split_values[i];
      }
    }
  }

  delete[] possible_split_values;
  delete[] num_deaths_right_child;
  delete[] delta_samples_at_risk_right_child;
  delete[] num_samples_right_child;
}

void TreeSurvival::findBestSplitValueLogRankUnordered(size_t nodeID, size_t varID, double& best_value,
    size_t& best_varID, double& best_logrank) {

  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];

  // Create possible split values
  double* factor_levels = new double[num_samples_node];
  size_t num_factor_levels = data->getAllValues(factor_levels, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Try next variable if all equal for this
  if (num_factor_levels < 2) {
    delete[] factor_levels;
    return;
  }

  // Number of possible splits is 2^num_levels
  size_t num_splits = (1ULL << num_factor_levels);

  // Compute logrank test statistic for each possible split
  // Split where all left (0) or all right (1) are excluded
  // The second half of numbers is just left/right switched the first half -> Exclude second half
  for (size_t local_splitID = 1; local_splitID < num_splits / 2; ++local_splitID) {

    // Compute overall splitID by shifting local factorIDs to global positions
    size_t splitID = 0;
    for (size_t j = 0; j < num_factor_levels; ++j) {
      if ((local_splitID & (1ULL << j))) {
        double level = factor_levels[j];
        size_t factorID = floor(level) - 1;
        splitID = splitID | (1ULL << factorID);
      }
    }

    // Initialize
    size_t* num_deaths_right_child = new size_t[num_timepoints]();
    size_t* delta_samples_at_risk_right_child = new size_t[num_timepoints]();
    size_t num_samples_right_child = 0;
    double numerator = 0;
    double denominator_squared = 0;

    // Count deaths in right child per timepoint
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      size_t survival_timeID = (*response_timepointIDs)[sampleID];
      double value = data->get_x(sampleID, varID);
      size_t factorID = floor(value) - 1;

      // If in right child, count
      // In right child, if bitwise splitID at position factorID is 1
      if ((splitID & (1ULL << factorID))) {
        ++num_samples_right_child;
        ++delta_samples_at_risk_right_child[survival_timeID];
        if (data->get_y(sampleID, 1) == 1) {
          ++num_deaths_right_child[survival_timeID];
        }
      }
    }

    // Stop if minimal node size reached
    size_t num_samples_left_child = num_samples_node - num_samples_right_child;
    if (num_samples_right_child < min_node_size || num_samples_left_child < min_node_size) {
      delete[] num_deaths_right_child;
      delete[] delta_samples_at_risk_right_child;
      continue;
    }

    // Compute logrank test statistic for this split
    size_t num_samples_at_risk_right_child = num_samples_right_child;
    for (size_t t = 0; t < num_timepoints; ++t) {
      if (num_samples_at_risk[t] < 2 || num_samples_at_risk_right_child < 1) {
        break;
      }

      if (num_deaths[t] > 0) {
        // Numerator and denominator for log-rank test, notation from Ishwaran et al.
        double di = static_cast<double>(num_deaths[t]);
        double di1 = static_cast<double>(num_deaths_right_child[t]);
        double Yi = static_cast<double>(num_samples_at_risk[t]);
        double Yi1 = static_cast<double>(num_samples_at_risk_right_child);
        numerator += di1 - Yi1 * (di / Yi);
        denominator_squared += (Yi1 / Yi) * (1.0 - Yi1 / Yi) * ((Yi - di) / (Yi - 1)) * di;
      }

      // Reduce number of samples at risk for next timepoint
      num_samples_at_risk_right_child -= delta_samples_at_risk_right_child[t];
    }
    double logrank = -1;
    if (denominator_squared != 0) {
      logrank = fabs(numerator / sqrt(denominator_squared));
    }

    // Regularization
    regularize(logrank, varID);

    if (logrank > best_logrank) {
      best_value = splitID;
      best_varID = varID;
      best_logrank = logrank;
    }

    delete[] num_deaths_right_child;
    delete[] delta_samples_at_risk_right_child;
  }

  delete[] factor_levels;
}

/////////////////////////////////////////////////////////////////////////////
void TreeSurvival::findBestSplitValueAUC(size_t nodeID, size_t varID, double& best_value, size_t& best_varID,
    double& best_auc) {

  // Create possible split values
  size_t num_node_samples = end_pos[nodeID] - start_pos[nodeID];
  double* possible_split_values = new double[num_node_samples];
  size_t num_split_values = data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Try next variable if all equal for this
  if (num_split_values < 2) {
    delete[] possible_split_values;
    return;
  }

  size_t num_splits = num_split_values - 1;
  size_t num_possible_pairs = num_node_samples * (num_node_samples - 1) / 2;

  // Initialize
  double* num_count = new double[num_splits];
  double* num_total = new double[num_splits];
  size_t* num_samples_left_child = new size_t[num_splits]();

  std::fill(num_count, num_count + num_splits, num_possible_pairs);
  std::fill(num_total, num_total + num_splits, num_possible_pairs);

  // For all pairs
  for (size_t k = start_pos[nodeID]; k < end_pos[nodeID]; ++k) {
    size_t sample_k = sampleIDs[k];
    double time_k = data->get_y(sample_k, 0);
    double status_k = data->get_y(sample_k, 1);
    double value_k = data->get_x(sample_k, varID);

    // Count samples in left node
    for (size_t i = 0; i < num_splits; ++i) {
      double split_value = possible_split_values[i];
      if (value_k <= split_value) {
        ++num_samples_left_child[i];
      }
    }

    for (size_t l = k + 1; l < end_pos[nodeID]; ++l) {
      size_t sample_l = sampleIDs[l];
      double time_l = data->get_y(sample_l, 0);
      double status_l = data->get_y(sample_l, 1);
      double value_l = data->get_x(sample_l, varID);

      // Compute split
      computeAucSplit(time_k, time_l, status_k, status_l, value_k, value_l, num_splits,
          possible_split_values, num_count, num_total);
    }
  }

  for (size_t i = 0; i < num_splits; ++i) {
    // Do not consider this split point if fewer than min_node_size samples in one node
    size_t num_samples_right_child = num_node_samples - num_samples_left_child[i];
    if (num_samples_left_child[i] < min_node_size || num_samples_right_child < min_node_size) {
      continue;
    } else {
      double auc = fabs((num_count[i] / 2) / num_total[i] - 0.5);

      // Regularization
      regularize(auc, varID);

      if (auc > best_auc) {
        best_value = (possible_split_values[i] + possible_split_values[i + 1]) / 2;
        best_varID = varID;
        best_auc = auc;

        // Use smaller value if average is numerically the same as the larger value
        if (best_value == possible_split_values[i + 1]) {
          best_value = possible_split_values[i];
        }
      }
    }
  }

  // Clean up
  delete[] possible_split_values;
  delete[] num_count;
  delete[] num_total;
  delete[] num_samples_left_child;
}

void TreeSurvival::computeAucSplit(double time_k, double time_l, double status_k, double status_l, double value_k,
    double value_l, size_t num_splits, double* possible_split_values, double* num_count, double* num_total) {

  bool ignore_pair = false;
  bool do_nothing = false;

  double value_smaller = 0;
  double value_larger = 0;
  double status_smaller = 0;

  if (time_k < time_l) {
    value_smaller = value_k;
    value_larger = value_l;
    status_smaller = status_k;
  } else if (time_l < time_k) {
    value_smaller = value_l;
    value_larger = value_k;
    status_smaller = status_l;
  } else {
    // Tie in survival time
    if (status_k == 0 || status_l == 0) {
      ignore_pair = true;
    } else {
      if (splitrule == AUC_IGNORE_TIES) {
        ignore_pair = true;
      } else {
        if (value_k == value_l) {
          // Tie in survival time and in covariate
          ignore_pair = true;
        } else {
          // Tie in survival time in covariate
          do_nothing = true;
        }
      }
    }
  }

  // Do not count if smaller time censored
  if (status_smaller == 0) {
    ignore_pair = true;
  }

  if (ignore_pair) {
    for (size_t i = 0; i < num_splits; ++i) {
      --num_count[i];
      --num_total[i];
    }
  } else if (do_nothing) {
    // Do nothing
  } else {
    for (size_t i = 0; i < num_splits; ++i) {
      double split_value = possible_split_values[i];

      if (value_smaller <= split_value && value_larger > split_value) {
        ++num_count[i];
      } else if (value_smaller > split_value && value_larger <= split_value) {
        --num_count[i];
      } else if (value_smaller <= split_value && value_larger <= split_value) {
        break;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
bool TreeSurvival::findBestSplitExtraTrees(size_t nodeID, size_t* possible_split_varIDs, size_t num_possible_split_varIDs) {

  double best_decrease = -1;
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  size_t best_varID = 0;
  double best_value = 0;

  computeDeathCounts(nodeID);

  // Stop if maximum node size or depth reached (will check again for each child node)
  if (num_samples_node <= min_node_size || (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth)) {
    computeSurvival(nodeID);
    return true;
  }

  // Stop early if no split possible
  if (num_samples_node >= 2 * min_node_size) {

    // For all possible split variables
    for (size_t i = 0; i < num_possible_split_varIDs; ++i) {
      size_t varID = possible_split_varIDs[i];

      // Find best split value, if ordered consider all values as split values, else all 2-partitions
      if (data->isOrderedVariable(varID)) {
        findBestSplitValueExtraTrees(nodeID, varID, best_value, best_varID, best_decrease);
      } else {
        findBestSplitValueExtraTreesUnordered(nodeID, varID, best_value, best_varID, best_decrease);
      }
    }
  }

  // Stop and save CHF if no good split found (this is terminal node).
  if (best_decrease < 0) {
    computeSurvival(nodeID);
    return true;
  } else {
    // If not terminal node save best values
    split_varIDs[nodeID] = best_varID;
    split_values[nodeID] = best_value;

    // Compute decrease of impurity for this node and add to variable importance if needed
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
      addImpurityImportance(nodeID, best_varID, best_decrease);
    }

    // Regularization
    saveSplitVarID(best_varID);

    return false;
  }
}

/////////////////////////////////////////////////////////////////////////////
void TreeSurvival::findBestSplitValueExtraTrees(size_t nodeID, size_t varID, double* best_value, size_t* best_varID,
    double* best_logrank) {

  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];

  // Get min/max values of covariate in node
  double min;
  double max;
  data->getMinMaxValues(min, max, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Try next variable if all equal for this
  if (min == max) {
    return;
  }

  // Create possible split values: Draw randomly between min and max
  double* possible_split_values = new double[num_random_splits];
  std::uniform_real_distribution<double> udist(min, max);
  for (size_t i = 0; i < num_random_splits; ++i) {
    possible_split_values[i] = udist(random_number_generator);
  }
  if (num_random_splits > 1) {
    std::sort(possible_split_values, possible_split_values + num_random_splits);
  }

  size_t num_splits = num_random_splits;

  // Initialize
  size_t* num_deaths_right_child = new size_t[num_splits * num_timepoints];
  size_t* delta_samples_at_risk_right_child = new size_t[num_splits * num_timepoints];
  size_t* num_samples_right_child = new size_t[num_splits];

  computeChildDeathCounts(nodeID, varID, possible_split_values, num_samples_right_child,
      delta_samples_at_risk_right_child, num_deaths_right_child, num_splits);

  // Compute logrank test for all splits and use best
  for (size_t i = 0; i < num_splits; ++i) {
    double numerator = 0;
    double denominator_squared = 0;

    // Stop if minimal node size reached
    size_t num_samples_left_child = num_samples_node - num_samples_right_child[i];
    if (num_samples_right_child[i] < min_node_size || num_samples_left_child < min_node_size) {
      continue;
    }

    // Compute logrank test statistic for this split
    size_t num_samples_at_risk_right_child = num_samples_right_child[i];
    for (size_t t = 0; t < num_timepoints; ++t) {
      if (num_samples_at_risk[t] < 2 || num_samples_at_risk_right_child < 1) {
        break;
      }

      if (num_deaths[t] > 0) {
        // Numerator and denominator for log-rank test, notation from Ishwaran et al.
        double di = static_cast<double>(num_deaths[t]);
        double di1 = static_cast<double>(num_deaths_right_child[i * num_timepoints + t]);
        double Yi = static_cast<double>(num_samples_at_risk[t]);
        double Yi1 = static_cast<double>(num_samples_at_risk_right_child);
        numerator += di1 - Yi1 * (di / Yi);
        denominator_squared += (Yi1 / Yi) * (1.0 - Yi1 / Yi) * ((Yi - di) / (Yi - 1)) * di;
      }

      // Reduce number of samples at risk for next timepoint
      num_samples_at_risk_right_child -= delta_samples_at_risk_right_child[i * num_timepoints + t];

    }
    double logrank = -1;
    if (denominator_squared != 0) {
      logrank = fabs(numerator / sqrt(denominator_squared));
    }

    // Regularization
    regularize(logrank, varID);

    if (logrank > *best_logrank) {
      *best_value = possible_split_values[i];
      *best_varID = varID;
      *best_logrank = logrank;
    }
  }

  // Clean up
  delete[] possible_split_values;
  delete[] num_deaths_right_child;
  delete[] delta_samples_at_risk_right_child;
  delete[] num_samples_right_child;
}

/////////////////////////////////////////////////////////////////////////////
void TreeSurvival::findBestSplitValueExtraTreesUnordered(size_t nodeID, size_t varID, double* best_value,
    size_t* best_varID, double* best_logrank) {

  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  size_t num_unique_values = data->getNumUniqueDataValues(varID);

  // Get all factor indices in node
  bool* factor_in_node = new bool[num_unique_values]();
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    size_t index = data->getIndex(sampleID, varID);
    factor_in_node[index] = true;
  }

  // Vector of indices in and out of node
  size_t* indices_in_node = new size_t[num_unique_values];
  size_t* indices_out_node = new size_t[num_unique_values];
  size_t indices_in_node_count = 0;
  size_t indices_out_node_count = 0;
  for (size_t i = 0; i < num_unique_values; ++i) {
    if (factor_in_node[i]) {
      indices_in_node[indices_in_node_count++] = i;
    } else {
      indices_out_node[indices_out_node_count++] = i;
    }
  }

  // Generate num_random_splits splits
  for (size_t i = 0; i < num_random_splits; ++i) {
    size_t* split_subset = new size_t[num_unique_values];
    size_t split_subset_count = 0;

    // Draw random subsets, sample all partitions with equal probability
    if (indices_in_node_count > 1) {
      size_t num_partitions = (2ULL << (indices_in_node_count - 1ULL)) - 2ULL; // 2^n-2 (don't allow full or empty)
      std::uniform_int_distribution<size_t> udist(1, num_partitions);
      size_t splitID_in_node = udist(random_number_generator);
      for (size_t j = 0; j < indices_in_node_count; ++j) {
        if ((splitID_in_node & (1ULL << j)) > 0) {
          split_subset[split_subset_count++] = indices_in_node[j];
        }
      }
    }
    if (indices_out_node_count > 1) {
      size_t num_partitions = (2ULL << (indices_out_node_count - 1ULL)) - 1ULL; // 2^n-1 (allow full or empty)
      std::uniform_int_distribution<size_t> udist(0, num_partitions);
      size_t splitID_out_node = udist(random_number_generator);
      for (size_t j = 0; j < indices_out_node_count; ++j) {
        if ((splitID_out_node & (1ULL << j)) > 0) {
          split_subset[split_subset_count++] = indices_out_node[j];
        }
      }
    }

    // Assign union of the two subsets to right child
    size_t splitID = 0;
    for (size_t j = 0; j < split_subset_count; ++j) {
      splitID |= 1ULL << split_subset[j];
    }

    // Initialize
    size_t* num_deaths_right_child = new size_t[num_timepoints]();
    size_t* delta_samples_at_risk_right_child = new size_t[num_timepoints]();
    size_t num_samples_right_child = 0;
    double numerator = 0;
    double denominator_squared = 0;

    // Count deaths in right child per timepoint
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      size_t survival_timeID = (*response_timepointIDs)[sampleID];
      double value = data->get_x(sampleID, varID);
      size_t factorID = floor(value) - 1;

      // If in right child, count
      // In right child, if bitwise splitID at position factorID is 1
      if ((splitID & (1ULL << factorID))) {
        ++num_samples_right_child;
        ++delta_samples_at_risk_right_child[survival_timeID];
        if (data->get_y(sampleID, 1) == 1) {
          ++num_deaths_right_child[survival_timeID];
        }
      }
    }

    // Stop if minimal node size reached
    size_t num_samples_left_child = num_samples_node - num_samples_right_child;
    if (num_samples_right_child < min_node_size || num_samples_left_child < min_node_size) {
      delete[] split_subset;
      delete[] num_deaths_right_child;
      delete[] delta_samples_at_risk_right_child;
      continue;
    }

    // Compute logrank test statistic for this split
    size_t num_samples_at_risk_right_child = num_samples_right_child;
    for (size_t t = 0; t < num_timepoints; ++t) {
      if (num_samples_at_risk[t] < 2 || num_samples_at_risk_right_child < 1) {
        break;
      }

      if (num_deaths[t] > 0) {
        // Numerator and denominator for log-rank test, notation from Ishwaran et al.
        double di = static_cast<double>(num_deaths[t]);
        double di1 = static_cast<double>(num_deaths_right_child[t]);
        double Yi = static_cast<double>(num_samples_at_risk[t]);
        double Yi1 = static_cast<double>(num_samples_at_risk_right_child);
        numerator += di1 - Yi1 * (di / Yi);
        denominator_squared += (Yi1 / Yi) * (1.0 - Yi1 / Yi) * ((Yi - di) / (Yi - 1)) * di;
      }

      // Reduce number of samples at risk for next timepoint
      num_samples_at_risk_right_child -= delta_samples_at_risk_right_child[t];
    }
    double logrank = -1;
    if (denominator_squared != 0) {
      logrank = fabs(numerator / sqrt(denominator_squared));
    }

    // Regularization
    regularize(logrank, varID);

    if (logrank > *best_logrank) {
      *best_value = splitID;
      *best_varID = varID;
      *best_logrank = logrank;
    }

    // Clean up
    delete[] split_subset;
    delete[] num_deaths_right_child;
    delete[] delta_samples_at_risk_right_child;
  }

  // Clean up
  delete[] factor_in_node;
  delete[] indices_in_node;
  delete[] indices_out_node;
}

void TreeSurvival::addImpurityImportance(size_t nodeID, size_t varID, double decrease) {

  // No variable importance for no split variables
  size_t tempvarID = data->getUnpermutedVarID(varID);

  // Subtract if corrected importance and permuted variable, else add
  if (importance_mode == IMP_GINI_CORRECTED && varID >= data->getNumCols()) {
    (*variable_importance)[tempvarID] -= decrease;
  } else {
    (*variable_importance)[tempvarID] += decrease;
  }
}
// namespace ranger
