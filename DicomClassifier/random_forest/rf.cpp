#include "rf.h"
#include <stdexcept>
#include "random_forest/globals.h"
#include "random_forest/ForestClassification.h"
#include "random_forest/utility.h"
#include "random_forest/Data.h"
#include <fstream>

using namespace std;

RF::RF(int trees = DEFAULT_NUM_TREE)
{
    std::unique_ptr<Forest> forest{};

    this->forest = ranger::make_unique<ForestClassification>();
    // Default Config to predict
    this->totalTrees = trees;
    this->mtry = 0;
    this->mode = MEM_DOUBLE;
    this->outprefix = "random_forest";
    this->default_seed = 0;
    this->predict_file = "";
    this->split_weights_file = "";
    this->depvarname = "LABEL";
    this->status_var_name = "";
    this->replacement = false;
    this->save_memory = false;
    this->predall = false;
    this->samplefraction = 0;
    this->holdout = false;
    this->reg_usedepth = false;
}

RF::~RF()
{}

void RF::setTrainData(double** data, int dataRows, int dataCols, int* target, int targetSize)
{
    this->filename = this->generateDataFormat(data, dataRows, dataCols, target, targetSize); //// Needs filename train
}

void RF::predict(double** data, int dataRows, int dataCols, int* target, int targetSize, bool showOutput)
{
    this->predict_file = this->generateDataFormat(data, dataRows, dataCols, target, targetSize); //// Needs filename train or test
    this->init(showOutput);
}

void RF::init(bool showOutput)
{
    if (showOutput) {
        this->forest->initCpp(depvarname, mode, filename, mtry, outprefix, totalTrees, &std::cout,
            default_seed, DEFAULT_NUM_THREADS, predict_file, DEFAULT_IMPORTANCE_MODE, DEFAULT_MIN_NODE_SIZE_CLASSIFICATION,
            split_weights_file, split_vars, status_var_name, replacement, cat_vars, save_memory,
            DEFAULT_SPLITRULE, weights_file, predall, samplefraction, DEFAULT_ALPHA,
            DEFAULT_MINPROP, holdout, DEFAULT_PREDICTIONTYPE, DEFAULT_NUM_RANDOM_SPLITS,
            DEFAULT_MAXDEPTH, reg_factor, reg_usedepth);
    }
    else {
        this->forest->initCpp(depvarname, mode, filename, mtry, outprefix, totalTrees, nullptr,
            default_seed, DEFAULT_NUM_THREADS, predict_file, DEFAULT_IMPORTANCE_MODE, DEFAULT_MIN_NODE_SIZE_CLASSIFICATION,
            split_weights_file, split_vars, status_var_name, replacement, cat_vars, save_memory,
            DEFAULT_SPLITRULE, weights_file, predall, samplefraction, DEFAULT_ALPHA,
            DEFAULT_MINPROP, holdout, DEFAULT_PREDICTIONTYPE, DEFAULT_NUM_RANDOM_SPLITS,
            DEFAULT_MAXDEPTH, reg_factor, reg_usedepth);
    }
}

void RF::setFile(string filename)
{
    this->filename = filename;
}

void RF::setPredictFile(string filename)
{
    this->predict_file = filename;
}

void RF::setDepVarName(string name)
{
    this->depvarname = name;
}

string RF::generateDataFormat(double** data, int dataRows, int dataCols, int* target, int targetSize)
{
    string filename = "temp.data";
    string delimiter = " ";

    if (dataRows != targetSize) {
        string message = "Data dimension and target size must be equal";
        throw runtime_error(message);
    }

    std::ofstream output_file(filename);

    // Write header
    for (int i = 0; i < dataCols; ++i) {
        output_file << "X" << i << delimiter;
    }
    output_file << "LABEL" << endl;

    // Write data
    for (int i = 0; i < dataRows; ++i) {
        for (int j = 0; j < dataCols; ++j) {
            output_file << data[i][j] << delimiter;
        }
        output_file << target[i] << "\n";
    }

    return filename;
}

