#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <memory>

#include "random_forest/globals.h"
#include "random_forest/ArgumentHandler.h"
#include "random_forest/ForestClassification.h"
#include "random_forest/ForestRegression.h"
#include "random_forest/ForestSurvival.h"
#include "random_forest/ForestProbability.h"
#include "random_forest/utility.h"
#include "random_forest/Data.h"

#include "dicom/DicomReader.h"

using namespace ranger;
using namespace std;

vector<vector<int>> getDataWithLabels(string labelPath, int **data, int width, int height) {
    vector<vector<int>> dataWithLabels;
    ifstream file(labelPath);
    string line = "";
    string delimiter = ",";
    vector<int> labels;

    // Read label values and push to vector
    while (getline(file, line))
    {
        size_t pos = 0;
        string token;
        while ((pos = line.find(delimiter)) != string::npos) {
            token = line.substr(0, pos);
            line.erase(0, pos + delimiter.length());
            labels.push_back(stoi(token));
        }
        labels.push_back(stoi(line));
    }
    // Close the File
    file.close();

    // Create matrix with data and labels
    for (int i = 0; i < height; ++i) {
        vector<int> dataset;
        for (int j = 0; j < width; ++j) {
            dataset.push_back(data[i][j]);
        }
        dataset.push_back(labels[i]);
        dataWithLabels.push_back(dataset);
    }

    return dataWithLabels;
}

int main () {
    DicomReader dicomObj("/home/adriana/dicom-classifier2/data/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm");
    int rows = dicomObj.getHeight();
    int cols = dicomObj.getWidth();

    int **data = dicomObj.getImageArray(12);

    const char *datafile = "dataset.csv";
    vector<vector<int>> dwl = getDataWithLabels("/home/adriana/dicom-classifier2/data/labels.csv", data, cols, rows);
    dicomObj.saveData(dwl, datafile, " ", true); // Random forest format needs space as delimiter and header

    // Create forest
    unique_ptr<ForestClassification> forest(new ForestClassification());

    // Config
    uint mtry = 0;
    MemoryMode mode = MEM_DOUBLE;
    string filename = datafile;
    string outprefix = "out_file";
    uint default_seed = 0;
    string predict_file = "";
    string split_weights_file = "";
    vector<string> split_vars;
    string status_var_name = "";
    bool replacement = false;
    vector<string> cat_vars;
    bool save_memory = false;
    string weights_file;
    bool predall = false;
    double samplefraction = 0;
    bool holdout = false;
    vector<double> reg_factor;
    bool reg_usedepth = false;

    forest->initCpp("LABEL", mode, filename, mtry, outprefix, DEFAULT_NUM_TREE, &cout,
                    default_seed, DEFAULT_NUM_THREADS, predict_file, DEFAULT_IMPORTANCE_MODE, DEFAULT_MIN_NODE_SIZE_CLASSIFICATION,
                    split_weights_file, split_vars, status_var_name, replacement, cat_vars, save_memory,
                    DEFAULT_SPLITRULE, weights_file, predall, samplefraction, DEFAULT_ALPHA,
                    DEFAULT_MINPROP, holdout, DEFAULT_PREDICTIONTYPE, DEFAULT_NUM_RANDOM_SPLITS,
                    DEFAULT_MAXDEPTH, reg_factor, reg_usedepth);

    forest->run(true, true);

    forest->writeOutput();
    forest->saveToFile();

    return 0;