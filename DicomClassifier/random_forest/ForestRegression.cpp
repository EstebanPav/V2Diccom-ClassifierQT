#include <algorithm>
#include <stdexcept>
#include <string>
#include <memory>

#include "random_forest/utility.h"
#include "ForestRegression.h"
#include "random_forest/TreeRegression.h"
#include "random_forest/Data.h"

namespace ranger {

// Modificación para usar punteros en lugar de std::vector
void ForestRegression::loadForest(size_t num_trees,
                                  size_t** forest_child_nodeIDs, size_t* forest_split_varIDs, double** forest_split_values,
                                  bool* is_ordered_variable) {

    this->num_trees = num_trees;
    data->setIsOrderedVariable(is_ordered_variable);

    // Create trees
    trees.reserve(num_trees); // Usando punteros en lugar de std::vector
    for (size_t i = 0; i < num_trees; ++i) {
        trees.push_back(
            std::make_unique<TreeRegression>(forest_child_nodeIDs[i], forest_split_varIDs[i], forest_split_values[i]));
    }

    // Create thread ranges
    equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
}

void ForestRegression::initInternal() {

    // If mtry not set, use floored square root of number of independent variables
    if (mtry == 0) {
        unsigned long temp = sqrt((double) num_independent_variables);
        mtry = std::max((unsigned long) 1, temp);
    }

    // Set minimal node size
    if (min_node_size == 0) {
        min_node_size = DEFAULT_MIN_NODE_SIZE_REGRESSION;
    }

    // Error if beta splitrule used with data outside of [0,1]
    if (splitrule == BETA && !prediction_mode) {
        for (size_t i = 0; i < num_samples; ++i) {
            double y = data->get_y(i, 0);
            if (y < 0 || y > 1) {
                throw std::runtime_error("Beta splitrule applicable to regression data with outcome between 0 and 1 only.");
            }
        }
    }

    // Sort data if memory saving mode
    if (!memory_saving_splitting) {
        data->sort();
    }
}

void ForestRegression::growInternal() {
    trees.reserve(num_trees); // Sin cambios
    for (size_t i = 0; i < num_trees; ++i) {
        trees.push_back(std::make_unique<TreeRegression>());
    }
}

// Modificación para usar punteros en lugar de std::vector
void ForestRegression::allocatePredictMemory() {
    size_t num_prediction_samples = data->getNumRows();
    if (predict_all || prediction_type == TERMINALNODES) {
        predictions = new double**[1];
        predictions[0] = new double*[num_prediction_samples];
        for (size_t i = 0; i < num_prediction_samples; ++i) {
            predictions[0][i] = new double[num_trees];
        }
    } else {
        predictions = new double**[1];
        predictions[0] = new double*[1];
        predictions[0][0] = new double[num_prediction_samples];
    }
}

void ForestRegression::predictInternal(size_t sample_idx) {
    if (predict_all || prediction_type == TERMINALNODES) {
        // Get all tree predictions
        for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
            if (prediction_type == TERMINALNODES) {
                predictions[0][sample_idx][tree_idx] = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
            } else {
                predictions[0][sample_idx][tree_idx] = getTreePrediction(tree_idx, sample_idx);
            }
        }
    } else {
        // Mean over trees
        double prediction_sum = 0;
        for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
            prediction_sum += getTreePrediction(tree_idx, sample_idx);
        }
        predictions[0][0][sample_idx] = prediction_sum / num_trees;
    }
}

// Modificación para usar punteros en lugar de std::vector
void ForestRegression::computePredictionErrorInternal() {

    // For each sample sum over trees where sample is OOB
    size_t* samples_oob_count = new size_t[num_samples]();
    predictions = new double**[1];
    predictions[0] = new double*[1];
    predictions[0][0] = new double[num_samples]();

    for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        for (size_t sample_idx = 0; sample_idx < trees[tree_idx]->getNumSamplesOob(); ++sample_idx) {
            size_t sampleID = trees[tree_idx]->getOobSampleIDs()[sample_idx];
            double value = getTreePrediction(tree_idx, sample_idx);

            predictions[0][0][sampleID] += value;
            ++samples_oob_count[sampleID];
        }
    }

    // MSE with predictions and true data
    size_t num_predictions = 0;
    overall_prediction_error = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        if (samples_oob_count[i] > 0) {
            ++num_predictions;
            predictions[0][0][i] /= (double) samples_oob_count[i];
            double predicted_value = predictions[0][0][i];
            double real_value = data->get_y(i, 0);
            overall_prediction_error += (predicted_value - real_value) * (predicted_value - real_value);
        } else {
            predictions[0][0][i] = NAN;
        }
    }

    overall_prediction_error /= (double) num_predictions;

    delete[] samples_oob_count;
}

// #nocov start
void ForestRegression::writeOutputInternal() {
    if (verbose_out) {
        *verbose_out << "Tree type:                         " << "Regression" << std::endl;
    }
}

void ForestRegression::writeConfusionFile() {

    // Open confusion file for writing
    std::string filename = output_prefix + ".confusion";
    std::ofstream outfile;
    outfile.open(filename, std::ios::out);
    if (!outfile.good()) {
        throw std::runtime_error("Could not write to confusion file: " + filename + ".");
    }

    // Write confusion to file
    outfile << "Overall OOB prediction error (MSE): " << overall_prediction_error << std::endl;

    outfile.close();
    if (verbose_out)
        *verbose_out << "Saved prediction error to file " << filename << "." << std::endl;
}

void ForestRegression::writePredictionFile() {

    // Open prediction file for writing
    std::string filename = output_prefix + ".prediction";
    std::ofstream outfile;
    outfile.open(filename, std::ios::out);
    if (!outfile.good()) {
        throw std::runtime_error("Could not write to prediction file: " + filename + ".");
    }

    // Write
    outfile << "Predictions: " << std::endl;
    if (predict_all) {
        for (size_t k = 0; k < num_trees; ++k) {
            outfile << "Tree " << k << ":" << std::endl;
            for (size_t i = 0; i < num_prediction_samples; ++i) {
                outfile << predictions[0][i][k] << std::endl;
            }
            outfile << std::endl;
        }
    } else {
        for (size_t i = 0; i < num_prediction_samples; ++i) {
            for (size_t j = 0; j < num_trees; ++j) {
                outfile << predictions[0][0][i] << std::endl;
            }
        }
    }

    if (verbose_out)
        *verbose_out << "Saved predictions to file " << filename << "." << std::endl;
}

void ForestRegression::saveToFileInternal(std::ofstream& outfile) {

    // Write num_variables
    outfile.write((char*) &num_independent_variables, sizeof(num_independent_variables));

    // Write treetype
    TreeType treetype = TREE_REGRESSION;
    outfile.write((char*) &treetype, sizeof(treetype));
}

void ForestRegression::loadFromFileInternal(std::ifstream& infile) {

    // Read number of variables
    size_t num_variables_saved;
    infile.read((char*) &num_variables_saved, sizeof(num_variables_saved));

    // Read treetype
    TreeType treetype;
    infile.read((char*) &treetype, sizeof(treetype));
    if (treetype != TREE_REGRESSION) {
        throw std::runtime_error("Wrong treetype. Loaded file is not a regression forest.");
    }

    for (size_t i = 0; i < num_trees; ++i) {

        // Read data
        std::vector<std::vector<size_t>> child_nodeIDs;
        readVector2D(child_nodeIDs, infile);
        std::vector<size_t> split_varIDs;
        readVector1D(split_varIDs, infile);
        std::vector<double> split_values;
        readVector1D(split_values, infile);

        // If dependent variable not in test data, throw error
        if (num_variables_saved != num_independent_variables) {
            throw std::runtime_error("Number of independent variables in data does not match with the loaded forest.");
        }

        // Create tree
        trees.push_back(std::make_unique<TreeRegression>(child_nodeIDs, split_varIDs, split_values));
    }
}

double ForestRegression::getTreePrediction(size_t tree_idx, size_t sample_idx) const {
    const auto& tree = dynamic_cast<const TreeRegression&>(*trees[tree_idx]);
    return tree.getPrediction(sample_idx);
}

size_t ForestRegression::getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const {
    const auto& tree = dynamic_cast<const TreeRegression&>(*trees[tree_idx]);
    return tree.getPredictionTerminalNodeID(sample_idx);
}

// #nocov end

}// namespace ranger
