/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#ifndef DATA_H_
#define DATA_H_

#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstring> // Para operaciones de cadenas

#include "globals.h"

namespace ranger {

class Data {
public:
    Data() : variable_names(nullptr), num_rows(0), num_rows_rounded(0), num_cols(0), snp_data(nullptr),
        num_cols_no_snp(0), externalData(false), index_data(nullptr), unique_data_values(nullptr),
        max_num_unique_values(0), is_ordered_variable(nullptr), permuted_sampleIDs(nullptr),
        snp_order(nullptr), order_snps(false) {}

    Data(const Data&) = delete;
    Data& operator=(const Data&) = delete;

    // Liberación de memoria manual
    virtual ~Data() {
        delete[] variable_names; // Cambio: Se eliminan vectores y se utilizan punteros
        delete[] index_data; // Cambio: Se eliminan vectores y se utilizan punteros
        for (size_t i = 0; i < num_cols; ++i) {
            delete[] unique_data_values[i]; // Cambio: Se eliminan vectores y se utilizan punteros
        }
        delete[] unique_data_values; // Cambio: Se eliminan vectores y se utilizan punteros
        delete[] is_ordered_variable; // Cambio: Se eliminan vectores y se utilizan punteros
        delete[] permuted_sampleIDs; // Cambio: Se eliminan vectores y se utilizan punteros
        for (size_t i = 0; i < num_cols_no_snp; ++i) {
            delete[] snp_order[i]; // Cambio: Se eliminan vectores y se utilizan punteros
        }
        delete[] snp_order; // Cambio: Se eliminan vectores y se utilizan punteros
    }

    virtual double get_x(size_t row, size_t col) const = 0;
    virtual double get_y(size_t row, size_t col) const = 0;

    size_t getVariableID(const std::string& variable_name) const;

    virtual void reserveMemory(size_t y_cols) = 0;

    virtual void set_x(size_t col, size_t row, double value, bool& error) = 0;
    virtual void set_y(size_t col, size_t row, double value, bool& error) = 0;

    void addSnpData(unsigned char* snp_data, size_t num_cols_snp);

    bool loadFromFile(std::string filename, std::string*& dependent_variable_names); // Cambio: vector a puntero
    bool loadFromFileWhitespace(std::ifstream& input_file, std::string header_line,
                                std::string*& dependent_variable_names); // Cambio: vector a puntero
    bool loadFromFileOther(std::ifstream& input_file, std::string header_line,
                           std::string*& dependent_variable_names, char seperator); // Cambio: vector a puntero

    void getAllValues(double*& all_values, size_t*& sampleIDs, size_t varID, size_t start,
                      size_t end) const; // Cambio: vector a puntero

    void getMinMaxValues(double& min, double& max, size_t*& sampleIDs, size_t varID, size_t start,
                         size_t end) const; // Cambio: vector a puntero

    size_t getIndex(size_t row, size_t col) const {
        size_t col_permuted = col;
        if (col >= num_cols) {
            col = getUnpermutedVarID(col);
            row = getPermutedSampleID(row);
        }

        if (col < num_cols_no_snp) {
            return index_data[col * num_rows + row]; // Cambio: Se eliminan vectores y se utilizan punteros
        } else {
            return getSnp(row, col, col_permuted); // Cambio: Se eliminan vectores y se utilizan punteros
        }
    }

    size_t getSnp(size_t row, size_t col, size_t col_permuted) const {
        size_t idx = (col - num_cols_no_snp) * num_rows_rounded + row;
        size_t result = ((snp_data[idx / 4] & mask[idx % 4]) >> offset[idx % 4]) - 1;

        if (result > 2) {
            result = 0;
        }

        if (order_snps) {
            if (col_permuted >= num_cols) {
                result = snp_order[col_permuted - 2 * num_cols_no_snp][result]; // Cambio: Se eliminan vectores y se utilizan punteros
            } else {
                result = snp_order[col - num_cols_no_snp][result]; // Cambio: Se eliminan vectores y se utilizan punteros
            }
        }
        return result;
    }

    double getUniqueDataValue(size_t varID, size_t index) const {
        if (varID >= num_cols) {
            varID = getUnpermutedVarID(varID);
        }

        if (varID < num_cols_no_snp) {
            return unique_data_values[varID][index]; // Cambio: Se eliminan vectores y se utilizan punteros
        } else {
            return (index);
        }
    }

    size_t getNumUniqueDataValues(size_t varID) const {
        if (varID >= num_cols) {
            varID = getUnpermutedVarID(varID);
        }

        if (varID < num_cols_no_snp) {
            return max_num_unique_values; // Cambio: Se eliminan vectores y se utilizan punteros
        } else {
            return (3);
        }
    }

    void sort();

    void orderSnpLevels(bool corrected_importance);

    const std::string* getVariableNames() const {
        return variable_names; // Cambio: Se eliminan vectores y se utilizan punteros
    }
    size_t getNumCols() const {
        return num_cols;
    }
    size_t getNumRows() const {
        return num_rows;
    }

    size_t getMaxNumUniqueValues() const {
        if (snp_data == nullptr || max_num_unique_values > 3) {
            return max_num_unique_values;
        } else {
            return 3;
        }
    }

    bool* getIsOrderedVariable() noexcept {
        return is_ordered_variable; // Cambio: Se eliminan vectores y se utilizan punteros
    }

    void setIsOrderedVariable(const std::string* unordered_variable_names, size_t size) {
        is_ordered_variable = new bool[num_cols]; // Cambio: Se eliminan vectores y se utilizan punteros
        std::fill(is_ordered_variable, is_ordered_variable + num_cols, true); // Cambio: Se eliminan vectores y se utilizan punteros
        for (size_t i = 0; i < size; ++i) {
            size_t varID = getVariableID(unordered_variable_names[i]);
            is_ordered_variable[varID] = false;
        }
    }

    void setIsOrderedVariable(bool* is_ordered_variable) {
        this->is_ordered_variable = is_ordered_variable; // Cambio: Se eliminan vectores y se utilizan punteros
    }

    bool isOrderedVariable(size_t varID) const {
        if (varID >= num_cols) {
            varID = getUnpermutedVarID(varID);
        }
        return is_ordered_variable[varID]; // Cambio: Se eliminan vectores y se utilizan punteros
    }

    void permuteSampleIDs(std::mt19937_64 random_number_generator) {
        permuted_sampleIDs = new size_t[num_rows]; // Cambio: Se eliminan vectores y se utilizan punteros
        std::iota(permuted_sampleIDs, permuted_sampleIDs + num_rows, 0); // Cambio: Se eliminan vectores y se utilizan punteros
        std::shuffle(permuted_sampleIDs, permuted_sampleIDs + num_rows, random_number_generator); // Cambio: Se eliminan vectores y se utilizan punteros
    }

    size_t getPermutedSampleID(size_t sampleID) const {
        return permuted_sampleIDs[sampleID]; // Cambio: Se eliminan vectores y se utilizan punteros
    }

    size_t getUnpermutedVarID(size_t varID) const {
        if (varID >= num_cols) {
            varID -= num_cols;
        }
        return varID;
    }

    const size_t* const* getSnpOrder() const {
        return snp_order; // Cambio: Se eliminan vectores y se utilizan punteros
    }

    void setSnpOrder(size_t** snp_order) {
        this->snp_order = snp_order; // Cambio: Se eliminan vectores y se utilizan punteros
        order_snps = true;
    }

protected:
    std::string* variable_names; // Cambio: Se eliminan vectores y se utilizan punteros
    size_t num_rows;
    size_t num_rows_rounded;
    size_t num_cols;

    unsigned char* snp_data;
    size_t num_cols_no_snp;

    bool externalData;

    size_t* index_data; // Cambio: Se eliminan vectores y se utilizan punteros
    double** unique_data_values; // Cambio: Se eliminan vectores y se utilizan punteros
    size_t max_num_unique_values;

    bool* is_ordered_variable; // Cambio: Se eliminan vectores y se utilizan punteros

    size_t* permuted_sampleIDs; // Cambio: Se eliminan vectores y se utilizan punteros

    size_t** snp_order; // Cambio: Se eliminan vectores y se utilizan punteros
    bool order_snps;
};

} // namespace ranger

#endif /* DATA_H_ */
