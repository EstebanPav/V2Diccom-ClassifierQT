/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <iterator>
#include <cstddef>  // For size_t

#include "Tree.h"
#include "random_forest/utility.h"

namespace ranger {

Tree::Tree() :
    mtry(0), num_samples(0), num_samples_oob(0), min_node_size(0), deterministic_varIDs(nullptr), split_select_weights(nullptr),
    case_weights(nullptr), manual_inbag(nullptr), oob_sampleIDs(nullptr), holdout(false), keep_inbag(false), data(nullptr),
    regularization_factor(nullptr), regularization_usedepth(false), split_varIDs_used(nullptr), variable_importance(nullptr),
    importance_mode(DEFAULT_IMPORTANCE_MODE), sample_with_replacement(true), sample_fraction(nullptr),
    memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), alpha(DEFAULT_ALPHA), minprop(DEFAULT_MINPROP),
    num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH), depth(0), last_left_nodeID(0) {
}

Tree::Tree(size_t** child_nodeIDs, size_t* split_varIDs, double* split_values, size_t child_nodeIDs_size,
    size_t split_varIDs_size, size_t split_values_size) :
    mtry(0), num_samples(0), num_samples_oob(0), min_node_size(0), deterministic_varIDs(nullptr), split_select_weights(nullptr),
    case_weights(nullptr), manual_inbag(nullptr), oob_sampleIDs(nullptr), holdout(false), keep_inbag(false), data(nullptr),
    regularization_factor(nullptr), regularization_usedepth(false), split_varIDs_used(nullptr), variable_importance(nullptr),
    importance_mode(DEFAULT_IMPORTANCE_MODE), sample_with_replacement(true), sample_fraction(nullptr),
    memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), alpha(DEFAULT_ALPHA), minprop(DEFAULT_MINPROP),
    num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH), depth(0), last_left_nodeID(0) {

    this->split_varIDs = new size_t[split_varIDs_size];
    std::copy(split_varIDs, split_varIDs + split_varIDs_size, this->split_varIDs);

    this->split_values = new double[split_values_size];
    std::copy(split_values, split_values + split_values_size, this->split_values);

    this->child_nodeIDs = new size_t*[child_nodeIDs_size];
    for (size_t i = 0; i < child_nodeIDs_size; ++i) {
        this->child_nodeIDs[i] = new size_t[/* appropriate size */]; // Define size if known
        std::copy(child_nodeIDs[i], child_nodeIDs[i] + /* appropriate size */, this->child_nodeIDs[i]);
    }
}

void Tree::init(const Data* data, uint mtry, size_t num_samples, uint seed, size_t* deterministic_varIDs, size_t deterministic_varIDs_size,
    double* split_select_weights, size_t split_select_weights_size, ImportanceMode importance_mode, uint min_node_size,
    bool sample_with_replacement, bool memory_saving_splitting, SplitRule splitrule, double* case_weights, size_t case_weights_size,
    size_t* manual_inbag, size_t manual_inbag_size, bool keep_inbag, double* sample_fraction, size_t sample_fraction_size, double alpha,
    double minprop, bool holdout, uint num_random_splits, uint max_depth, double* regularization_factor, size_t regularization_factor_size,
    bool regularization_usedepth, bool* split_varIDs_used, size_t split_varIDs_used_size) {

    this->data = data;
    this->mtry = mtry;
    this->num_samples = num_samples;
    this->memory_saving_splitting = memory_saving_splitting;

    // Create root node, assign bootstrap sample and oob samples
    this->child_nodeIDs = new size_t*[2]; // Initialize with size 2 or according to requirements
    this->child_nodeIDs[0] = new size_t[/* appropriate size */]; // Define size if known
    this->child_nodeIDs[1] = new size_t[/* appropriate size */]; // Define size if known
    createEmptyNode();

    // Initialize random number generator and set seed
    random_number_generator.seed(seed);

    this->deterministic_varIDs = new size_t[deterministic_varIDs_size];
    std::copy(deterministic_varIDs, deterministic_varIDs + deterministic_varIDs_size, this->deterministic_varIDs);

    this->split_select_weights = new double[split_select_weights_size];
    std::copy(split_select_weights, split_select_weights + split_select_weights_size, this->split_select_weights);

    this->importance_mode = importance_mode;
    this->min_node_size = min_node_size;
    this->sample_with_replacement = sample_with_replacement;
    this->splitrule = splitrule;

    this->case_weights = new double[case_weights_size];
    std::copy(case_weights, case_weights + case_weights_size, this->case_weights);

    this->manual_inbag = new size_t[manual_inbag_size];
    std::copy(manual_inbag, manual_inbag + manual_inbag_size, this->manual_inbag);

    this->keep_inbag = keep_inbag;

    this->sample_fraction = new double[sample_fraction_size];
    std::copy(sample_fraction, sample_fraction + sample_fraction_size, this->sample_fraction);

    this->holdout = holdout;
    this->alpha = alpha;
    this->minprop = minprop;
    this->num_random_splits = num_random_splits;
    this->max_depth = max_depth;

    this->regularization_factor = new double[regularization_factor_size];
    std::copy(regularization_factor, regularization_factor + regularization_factor_size, this->regularization_factor);

    this->regularization_usedepth = regularization_usedepth;

    this->split_varIDs_used = new bool[split_varIDs_used_size];
    std::copy(split_varIDs_used, split_varIDs_used + split_varIDs_used_size, this->split_varIDs_used);

    // Regularization
    if (regularization_factor_size > 0) {
        regularization = true;
    } else {
        regularization = false;
    }
}
} // namespace ranger

//INICIO2
#include <cmath>      // For floor
#include <cstring>    // For std::memset

void Tree::grow(double* variable_importance, size_t variable_importance_size) {
    // Allocate memory for tree growing
    allocateMemory();

    this->variable_importance = variable_importance;

    // Bootstrap, dependent if weighted or not and with or without replacement
    if (case_weights && !case_weights->empty()) {
        if (sample_with_replacement) {
            bootstrapWeighted();
        } else {
            bootstrapWithoutReplacementWeighted();
        }
    } else if (sample_fraction && sample_fraction_size > 1) {
        if (sample_with_replacement) {
            bootstrapClassWise();
        } else {
            bootstrapWithoutReplacementClassWise();
        }
    } else if (manual_inbag && !manual_inbag_size) {
        setManualInbag();
    } else {
        if (sample_with_replacement) {
            bootstrap();
        } else {
            bootstrapWithoutReplacement();
        }
    }

    // Init start and end positions
    start_pos[0] = 0;
    end_pos[0] = sampleIDs_size;  // Adjusted to work with size_t

    // While not all nodes terminal, split next node
    size_t num_open_nodes = 1;
    size_t i = 0;
    depth = 0;
    while (num_open_nodes > 0) {
        // Split node
        bool is_terminal_node = splitNode(i);
        if (is_terminal_node) {
            --num_open_nodes;
        } else {
            ++num_open_nodes;
            if (i >= last_left_nodeID) {
                // If new level, increase depth
                // (left_node saves left-most node in current level, new level reached if that node is splitted)
                last_left_nodeID = split_varIDs_size - 2;
                ++depth;
            }
        }
        ++i;
    }

    // Delete sampleID vector to save memory
    delete[] sampleIDs;
    sampleIDs = nullptr;
    sampleIDs_size = 0;

    cleanUpInternal();
}

void Tree::predict(const Data* prediction_data, bool oob_prediction) {

    size_t num_samples_predict;
    if (oob_prediction) {
        num_samples_predict = num_samples_oob;
    } else {
        num_samples_predict = prediction_data->getNumRows();
    }

    // Resize prediction_terminal_nodeIDs
    delete[] prediction_terminal_nodeIDs;
    prediction_terminal_nodeIDs = new size_t[num_samples_predict];
    std::memset(prediction_terminal_nodeIDs, 0, num_samples_predict * sizeof(size_t));

    // For each sample start in root, drop down the tree and return final value
    for (size_t i = 0; i < num_samples_predict; ++i) {
        size_t sample_idx;
        if (oob_prediction) {
            sample_idx = oob_sampleIDs[i];
        } else {
            sample_idx = i;
        }
        size_t nodeID = 0;
        while (true) {

            // Break if terminal node
            if (child_nodeIDs[0][nodeID] == 0 && child_nodeIDs[1][nodeID] == 0) {
                break;
            }

            // Move to child
            size_t split_varID = split_varIDs[nodeID];

            double value = prediction_data->get_x(sample_idx, split_varID);
            if (prediction_data->isOrderedVariable(split_varID)) {
                if (value <= split_values[nodeID]) {
                    // Move to left child
                    nodeID = child_nodeIDs[0][nodeID];
                } else {
                    // Move to right child
                    nodeID = child_nodeIDs[1][nodeID];
                }
            } else {
                size_t factorID = static_cast<size_t>(std::floor(value)) - 1;
                size_t splitID = static_cast<size_t>(std::floor(split_values[nodeID]));

                // Left if 0 found at position factorID
                if (!(splitID & (1ULL << factorID))) {
                    // Move to left child
                    nodeID = child_nodeIDs[0][nodeID];
                } else {
                    // Move to right child
                    nodeID = child_nodeIDs[1][nodeID];
                }
            }
        }

        prediction_terminal_nodeIDs[i] = nodeID;
    }
}

//FIN2
//INICIO 3
void Tree::computePermutationImportance(double* forest_importance, double* forest_variance,
    double* forest_importance_casewise, size_t num_independent_variables, size_t num_samples_oob) {

    // Compute normal prediction accuracy for each tree. Predictions already computed..
    double accuracy_normal;
    double* prederr_normal_casewise = new double[num_samples_oob];
    double* prederr_shuf_casewise = new double[num_samples_oob];
    if (importance_mode == IMP_PERM_CASEWISE) {
        std::memset(prederr_normal_casewise, 0, num_samples_oob * sizeof(double));
        std::memset(prederr_shuf_casewise, 0, num_samples_oob * sizeof(double));
        accuracy_normal = computePredictionAccuracyInternal(prederr_normal_casewise);
    } else {
        accuracy_normal = computePredictionAccuracyInternal(nullptr);
    }

    delete[] prediction_terminal_nodeIDs;
    prediction_terminal_nodeIDs = new size_t[num_samples_oob];
    std::memset(prediction_terminal_nodeIDs, 0, num_samples_oob * sizeof(size_t));

    // Reserve space for permutations, initialize with oob_sampleIDs
    size_t* permutations = new size_t[num_samples_oob];
    std::memcpy(permutations, oob_sampleIDs, num_samples_oob * sizeof(size_t));

    // Randomly permute for all independent variables
    for (size_t i = 0; i < num_independent_variables; ++i) {

        // Permute and compute prediction accuracy again for this permutation and save difference
        permuteAndPredictOobSamples(i, permutations);
        double accuracy_permuted;
        if (importance_mode == IMP_PERM_CASEWISE) {
            accuracy_permuted = computePredictionAccuracyInternal(prederr_shuf_casewise);
            for (size_t j = 0; j < num_samples_oob; ++j) {
                size_t pos = i * num_samples + oob_sampleIDs[j];
                forest_importance_casewise[pos] += prederr_shuf_casewise[j] - prederr_normal_casewise[j];
            }
        } else {
            accuracy_permuted = computePredictionAccuracyInternal(nullptr);
        }

        double accuracy_difference = accuracy_normal - accuracy_permuted;
        forest_importance[i] += accuracy_difference;

        // Compute variance
        if (importance_mode == IMP_PERM_BREIMAN) {
            forest_variance[i] += accuracy_difference * accuracy_difference;
        } else if (importance_mode == IMP_PERM_LIAW) {
            forest_variance[i] += accuracy_difference * accuracy_difference * num_samples_oob;
        }
    }

    delete[] prederr_normal_casewise;
    delete[] prederr_shuf_casewise;
    delete[] permutations;
}

void Tree::appendToFile(std::ofstream& file) {

    // Save general fields
    saveMatrix(child_nodeIDs, file);
    saveArray(split_varIDs, split_varIDs_size, file);
    saveArray(split_values, split_values_size, file);

    // Call special functions for subclasses to save special fields.
    appendToFileInternal(file);
}

void Tree::createPossibleSplitVarSubset(size_t* result, size_t& result_size) {

    size_t num_vars = data->getNumCols();

    // For corrected Gini importance add dummy variables
    if (importance_mode == IMP_GINI_CORRECTED) {
        num_vars += data->getNumCols();
    }

    // Randomly add non-deterministic variables (according to weights if needed)
    if (split_select_weights == nullptr || split_select_weights_size == 0) {
        if (deterministic_varIDs == nullptr || deterministic_varIDs_size == 0) {
            drawWithoutReplacement(result, result_size, num_vars, mtry);
        } else {
            drawWithoutReplacementSkip(result, result_size, num_vars, deterministic_varIDs, deterministic_varIDs_size, mtry);
        }
    } else {
        drawWithoutReplacementWeighted(result, result_size, num_vars, mtry, split_select_weights, split_select_weights_size);
    }

    // Always use deterministic variables
    size_t* tmp_result = new size_t[result_size + deterministic_varIDs_size];
    std::memcpy(tmp_result, result, result_size * sizeof(size_t));
    std::memcpy(tmp_result + result_size, deterministic_varIDs, deterministic_varIDs_size * sizeof(size_t));
    result_size += deterministic_varIDs_size;
    delete[] result;
    result = tmp_result;
}

bool Tree::splitNode(size_t nodeID) {

    // Select random subset of variables to possibly split at
    size_t* possible_split_varIDs = new size_t[num_independent_variables];
    size_t possible_split_varIDs_size;
    createPossibleSplitVarSubset(possible_split_varIDs, possible_split_varIDs_size);

    // Call subclass method, sets split_varIDs and split_values
    bool stop = splitNodeInternal(nodeID, possible_split_varIDs, possible_split_varIDs_size);
    if (stop) {
        // Terminal node
        delete[] possible_split_varIDs;
        return true;
    }

    size_t split_varID = split_varIDs[nodeID];
    double split_value = split_values[nodeID];

    // Save non-permuted variable for prediction
    split_varIDs[nodeID] = data->getUnpermutedVarID(split_varID);

    // Create child nodes
    size_t left_child_nodeID = split_varIDs_size;
    child_nodeIDs[0] = new size_t[left_child_nodeID + 1];
    child_nodeIDs[1] = new size_t[left_child_nodeID + 1];
    child_nodeIDs[0][nodeID] = left_child_nodeID;
    createEmptyNode();
    start_pos[left_child_nodeID] = start_pos[nodeID];

    size_t right_child_nodeID = split_varIDs_size + 1;
    child_nodeIDs[1][nodeID] = right_child_nodeID;
    createEmptyNode();
    start_pos[right_child_nodeID] = end_pos[nodeID];

    // For each sample in node, assign to left or right child
    if (data->isOrderedVariable(split_varID)) {
        // Ordered: left is <= splitval and right is > splitval
        size_t pos = start_pos[nodeID];
        while (pos < start_pos[right_child_nodeID]) {
            size_t sampleID = sampleIDs[pos];
            if (data->get_x(sampleID, split_varID) <= split_value) {
                // If going to left, do nothing
                ++pos;
            } else {
                // If going to right, move to right end
                --start_pos[right_child_nodeID];
                std::swap(sampleIDs[pos], sampleIDs[start_pos[right_child_nodeID]]);
            }
        }
    } else {
        // Unordered: If bit at position is 1 -> right, 0 -> left
        size_t pos = start_pos[nodeID];
        while (pos < start_pos[right_child_nodeID]) {
            size_t sampleID = sampleIDs[pos];
            double level = data->get_x(sampleID, split_varID);
            size_t factorID = static_cast<size_t>(std::floor(level)) - 1;
            size_t splitID = static_cast<size_t>(std::floor(split_value));

            // Left if 0 found at position factorID
            if (!(splitID & (1ULL << factorID))) {
                // If going to left, do nothing
                ++pos;
            } else {
                // If going to right, move to right end
                --start_pos[right_child_nodeID];
                std::swap(sampleIDs[pos], sampleIDs[start_pos[right_child_nodeID]]);
            }
        }
    }

    // End position of left child is start position of right child
    end_pos[left_child_nodeID] = start_pos[right_child_nodeID];
    end_pos[right_child_nodeID] = end_pos[nodeID];

    delete[] possible_split_varIDs;
    // No terminal node
    return false;
}

//FIN 3
//INICIO 4
void Tree::createEmptyNode() {
    size_t new_size = num_nodes + 1; // Assuming num_nodes keeps track of the number of nodes

    // Reallocate memory for arrays
    size_t* new_split_varIDs = new size_t[new_size];
    double* new_split_values = new double[new_size];
    size_t** new_child_nodeIDs = new size_t*[2];
    new_child_nodeIDs[0] = new size_t[new_size];
    new_child_nodeIDs[1] = new size_t[new_size];
    size_t* new_start_pos = new size_t[new_size];
    size_t* new_end_pos = new size_t[new_size];

    // Copy old data to new arrays
    std::memcpy(new_split_varIDs, split_varIDs, num_nodes * sizeof(size_t));
    std::memcpy(new_split_values, split_values, num_nodes * sizeof(double));
    std::memcpy(new_child_nodeIDs[0], child_nodeIDs[0], num_nodes * sizeof(size_t));
    std::memcpy(new_child_nodeIDs[1], child_nodeIDs[1], num_nodes * sizeof(size_t));
    std::memcpy(new_start_pos, start_pos, num_nodes * sizeof(size_t));
    std::memcpy(new_end_pos, end_pos, num_nodes * sizeof(size_t));

    // Add new elements
    new_split_varIDs[num_nodes] = 0;
    new_split_values[num_nodes] = 0;
    new_child_nodeIDs[0][num_nodes] = 0;
    new_child_nodeIDs[1][num_nodes] = 0;
    new_start_pos[num_nodes] = 0;
    new_end_pos[num_nodes] = 0;

    // Update pointers and delete old arrays
    delete[] split_varIDs;
    delete[] split_values;
    delete[] child_nodeIDs[0];
    delete[] child_nodeIDs[1];
    delete[] start_pos;
    delete[] end_pos;

    split_varIDs = new_split_varIDs;
    split_values = new_split_values;
    child_nodeIDs[0] = new_child_nodeIDs[0];
    child_nodeIDs[1] = new_child_nodeIDs[1];
    start_pos = new_start_pos;
    end_pos = new_end_pos;
    num_nodes++;

    createEmptyNodeInternal();
}

size_t Tree::dropDownSamplePermuted(size_t permuted_varID, size_t sampleID, size_t permuted_sampleID) {

    // Start in root and drop down
    size_t nodeID = 0;
    while (child_nodeIDs[0][nodeID] != 0 || child_nodeIDs[1][nodeID] != 0) {

        // Permute if variable is permutation variable
        size_t split_varID = split_varIDs[nodeID];
        size_t sampleID_final = sampleID;
        if (split_varID == permuted_varID) {
            sampleID_final = permuted_sampleID;
        }

        // Move to child
        double value = data->get_x(sampleID_final, split_varID);
        if (data->isOrderedVariable(split_varID)) {
            if (value <= split_values[nodeID]) {
                // Move to left child
                nodeID = child_nodeIDs[0][nodeID];
            } else {
                // Move to right child
                nodeID = child_nodeIDs[1][nodeID];
            }
        } else {
            size_t factorID = floor(value) - 1;
            size_t splitID = floor(split_values[nodeID]);

            // Left if 0 found at position factorID
            if (!(splitID & (1ULL << factorID))) {
                // Move to left child
                nodeID = child_nodeIDs[0][nodeID];
            } else {
                // Move to right child
                nodeID = child_nodeIDs[1][nodeID];
            }
        }
    }
    return nodeID;
}

void Tree::permuteAndPredictOobSamples(size_t permuted_varID, size_t* permutations) {

    // Permute OOB sample
    std::shuffle(permutations, permutations + num_samples_oob, random_number_generator);

    // For each sample, drop down the tree and add prediction
    for (size_t i = 0; i < num_samples_oob; ++i) {
        size_t nodeID = dropDownSamplePermuted(permuted_varID, oob_sampleIDs[i], permutations[i]);
        prediction_terminal_nodeIDs[i] = nodeID;
    }
}

void Tree::bootstrap() {

    // Use fraction (default 63.21%) of the samples
    size_t num_samples_inbag = static_cast<size_t>(num_samples * (*sample_fraction)[0]);

    // Reserve space, reserve a little more to be safe
    size_t* new_sampleIDs = new size_t[num_samples_inbag];
    size_t* new_oob_sampleIDs = new size_t[num_samples * (exp(-(*sample_fraction)[0]) + 0.1)];
    size_t* new_inbag_counts = new size_t[num_samples]();

    std::uniform_int_distribution<size_t> unif_dist(0, num_samples - 1);

    // Start with all samples OOB
    std::memset(new_inbag_counts, 0, num_samples * sizeof(size_t));

    // Draw num_samples samples with replacement (num_samples_inbag out of n) as inbag and mark as not OOB
    size_t sample_inbag_count = 0;
    for (size_t s = 0; s < num_samples_inbag; ++s) {
        size_t draw = unif_dist(random_number_generator);
        new_sampleIDs[sample_inbag_count++] = draw;
        ++new_inbag_counts[draw];
    }

    // Save OOB samples
    size_t oob_sample_count = 0;
    for (size_t s = 0; s < num_samples; ++s) {
        if (new_inbag_counts[s] == 0) {
            new_oob_sampleIDs[oob_sample_count++] = s;
        }
    }
    num_samples_oob = oob_sample_count;

    if (!keep_inbag) {
        delete[] new_inbag_counts;
    }

    delete[] sampleIDs;
    delete[] oob_sampleIDs;
    sampleIDs = new_sampleIDs;
    oob_sampleIDs = new_oob_sampleIDs;
}

//FIN 4
//INICIO 5
void Tree::bootstrapWeighted() {

    // Use fraction (default 63.21%) of the samples
    size_t num_samples_inbag = (size_t)num_samples * (*sample_fraction)[0];

    // Allocate memory for sampleIDs and oob_sampleIDs
    size_t* new_sampleIDs = new size_t[num_samples_inbag];
    size_t* new_oob_sampleIDs = new size_t[num_samples * (exp(-(*sample_fraction)[0]) + 0.1)];
    size_t* new_inbag_counts = new size_t[num_samples]();
    
    std::discrete_distribution<> weighted_dist(case_weights->begin(), case_weights->end());

    // Start with all samples OOB
    std::memset(new_inbag_counts, 0, num_samples * sizeof(size_t));

    // Draw num_samples samples with replacement (n out of n) as inbag and mark as not OOB
    size_t sample_inbag_count = 0;
    for (size_t s = 0; s < num_samples_inbag; ++s) {
        size_t draw = weighted_dist(random_number_generator);
        new_sampleIDs[sample_inbag_count++] = draw;
        ++new_inbag_counts[draw];
    }

    // Save OOB samples. In holdout mode these are the cases with 0 weight.
    size_t oob_sample_count = 0;
    if (holdout) {
        for (size_t s = 0; s < case_weights->size(); ++s) {
            if ((*case_weights)[s] == 0) {
                new_oob_sampleIDs[oob_sample_count++] = s;
            }
        }
    } else {
        for (size_t s = 0; s < num_samples; ++s) {
            if (new_inbag_counts[s] == 0) {
                new_oob_sampleIDs[oob_sample_count++] = s;
            }
        }
    }
    num_samples_oob = oob_sample_count;

    // Clean up old data
    delete[] sampleIDs;
    delete[] oob_sampleIDs;
    delete[] inbag_counts;

    // Update pointers
    sampleIDs = new_sampleIDs;
    oob_sampleIDs = new_oob_sampleIDs;
    inbag_counts = new_inbag_counts;
    
    if (!keep_inbag) {
        // inbag_counts are already cleared by deletion
    }
}

void Tree::bootstrapWithoutReplacement() {

    // Use fraction (default 63.21%) of the samples
    size_t num_samples_inbag = (size_t)num_samples * (*sample_fraction)[0];

    // Allocate memory for sampleIDs and oob_sampleIDs
    size_t* new_sampleIDs = new size_t[num_samples];
    size_t* new_oob_sampleIDs = new size_t[num_samples];
    
    shuffleAndSplit(new_sampleIDs, new_oob_sampleIDs, num_samples, num_samples_inbag, random_number_generator);

    // Count OOB samples
    size_t oob_sample_count = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        if (new_sampleIDs[i] == 0) {
            new_oob_sampleIDs[oob_sample_count++] = i;
        }
    }
    num_samples_oob = oob_sample_count;

    if (keep_inbag) {
        // All observations are 0 or 1 times inbag
        size_t* new_inbag_counts = new size_t[num_samples];
        std::memset(new_inbag_counts, 1, num_samples * sizeof(size_t));
        for (size_t i = 0; i < num_samples_oob; ++i) {
            new_inbag_counts[new_oob_sampleIDs[i]] = 0;
        }
        delete[] inbag_counts;
        inbag_counts = new_inbag_counts;
    }

    // Clean up old data
    delete[] sampleIDs;
    delete[] oob_sampleIDs;

    // Update pointers
    sampleIDs = new_sampleIDs;
    oob_sampleIDs = new_oob_sampleIDs;
}

void Tree::bootstrapWithoutReplacementWeighted() {

    // Use fraction (default 63.21%) of the samples
    size_t num_samples_inbag = (size_t)num_samples * (*sample_fraction)[0];

    // Allocate memory for sampleIDs and inbag_counts
    size_t* new_sampleIDs = new size_t[num_samples_inbag];
    size_t* new_inbag_counts = new size_t[num_samples]();
    size_t* new_oob_sampleIDs = new size_t[num_samples];
    
    drawWithoutReplacementWeighted(new_sampleIDs, random_number_generator, num_samples - 1, num_samples_inbag, *case_weights);

    // Count inbag occurrences
    for (size_t i = 0; i < num_samples_inbag; ++i) {
        new_inbag_counts[new_sampleIDs[i]] = 1;
    }

    // Save OOB samples
    size_t oob_sample_count = 0;
    if (holdout) {
        for (size_t s = 0; s < case_weights->size(); ++s) {
            if ((*case_weights)[s] == 0) {
                new_oob_sampleIDs[oob_sample_count++] = s;
            }
        }
    } else {
        for (size_t s = 0; s < num_samples; ++s) {
            if (new_inbag_counts[s] == 0) {
                new_oob_sampleIDs[oob_sample_count++] = s;
            }
        }
    }
    num_samples_oob = oob_sample_count;

    // Clean up old data
    delete[] sampleIDs;
    delete[] oob_sampleIDs;
    delete[] inbag_counts;

    // Update pointers
    sampleIDs = new_sampleIDs;
    oob_sampleIDs = new_oob_sampleIDs;
    inbag_counts = new_inbag_counts;
}

void Tree::bootstrapClassWise() {
    // Empty on purpose (virtual function only implemented in classification and probability)
}

void Tree::bootstrapWithoutReplacementClassWise() {
    // Empty on purpose (virtual function only implemented in classification and probability)
}

void Tree::setManualInbag() {
    // Allocate memory for sampleIDs and inbag_counts
    size_t* new_sampleIDs = new size_t[manual_inbag->size()];
    size_t* new_inbag_counts = new size_t[num_samples]();
    size_t* new_oob_sampleIDs = new size_t[num_samples];
    
    size_t sample_count = 0;
    size_t oob_count = 0;

    for (size_t i = 0; i < manual_inbag->size(); ++i) {
        size_t inbag_count = (*manual_inbag)[i];
        if (inbag_count > 0) {
            for (size_t j = 0; j < inbag_count; ++j) {
                new_sampleIDs[sample_count++] = i;
            }
            new_inbag_counts[i] = inbag_count;
        } else {
            new_oob_sampleIDs[oob_count++] = i;
        }
    }
    num_samples_oob = oob_count;

    // Shuffle samples
    std::shuffle(new_sampleIDs, new_sampleIDs + sample_count, random_number_generator);

    // Clean up old data
    delete[] sampleIDs;
    delete[] oob_sampleIDs;
    delete[] inbag_counts;

    // Update pointers
    sampleIDs = new_sampleIDs;
    oob_sampleIDs = new_oob_sampleIDs;
    inbag_counts = new_inbag_counts;

    if (!keep_inbag) {
        // inbag_counts are already cleared by deletion
    }
}

