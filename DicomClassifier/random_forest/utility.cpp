/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/
#include <math.h>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <fstream>
#include <cmath>
#include <limits>
#include <cstring> // For memcpy

#include "utility.h"
#include "globals.h"
#include "Data.h"

namespace ranger {

void equalSplit(uint* result, uint start, uint end, uint num_parts, uint& size) {
    size = 0;
    
    // Return range if only 1 part
    if (num_parts == 1) {
        result[size++] = start;
        result[size++] = end + 1;
        return;
    }

    // Return range from start to end+1 if more parts than elements
    if (num_parts > end - start + 1) {
        for (uint i = start; i <= end + 1; ++i) {
            result[size++] = i;
        }
        return;
    }

    uint length = (end - start + 1);
    uint part_length_short = length / num_parts;
    uint part_length_long = (uint) ceil(length / ((double) num_parts));
    uint cut_pos = length % num_parts;

    // Add long ranges
    for (uint i = start; i < start + cut_pos * part_length_long; i = i + part_length_long) {
        result[size++] = i;
    }

    // Add short ranges
    for (uint i = start + cut_pos * part_length_long; i <= end + 1; i = i + part_length_short) {
        result[size++] = i;
    }
}

void loadDoubleVectorFromFile(double*& result, size_t& size, std::string filename) { // #nocov start
    // Open input file
    std::ifstream input_file;
    input_file.open(filename);
    if (!input_file.good()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Read the first line, ignore the rest
    std::string line;
    getline(input_file, line);
    std::stringstream line_stream(line);
    double token;

    // Calculate the number of tokens
    size_t count = 0;
    while (line_stream >> token) {
        ++count;
    }

    // Allocate memory for result
    result = new double[count];
    size = count;

    // Reset the stream to read the tokens again
    input_file.clear();
    input_file.seekg(0, std::ios::beg);
    getline(input_file, line);
    line_stream.str(line);

    // Read the tokens into result
    count = 0;
    while (line_stream >> token) {
        result[count++] = token;
    }
} // #nocov end

void drawWithoutReplacement(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator, size_t max, size_t num_samples) {
    if (num_samples < max / 10) {
        drawWithoutReplacementSimple(result, result_size, random_number_generator, max, num_samples);
    } else {
        // drawWithoutReplacementKnuth(result, random_number_generator, max, skip, num_samples);
        drawWithoutReplacementFisherYates(result, result_size, random_number_generator, max, num_samples);
    }
}

void drawWithoutReplacementSkip(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator, size_t max, const size_t* skip, size_t skip_size, size_t num_samples) {
    if (num_samples < max / 10) {
        drawWithoutReplacementSimple(result, result_size, random_number_generator, max, skip, skip_size, num_samples);
    } else {
        // drawWithoutReplacementKnuth(result, random_number_generator, max, skip, num_samples);
        drawWithoutReplacementFisherYates(result, result_size, random_number_generator, max, skip, skip_size, num_samples);
    }
}

void drawWithoutReplacementSimple(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator, size_t max, size_t num_samples) {
    result = new size_t[num_samples];
    result_size = num_samples;

    // Set all to not selected
    bool* temp = new bool[max];
    std::fill(temp, temp + max, false);

    std::uniform_int_distribution<size_t> unif_dist(0, max - 1);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t draw;
        do {
            draw = unif_dist(random_number_generator);
        } while (temp[draw]);
        temp[draw] = true;
        result[i] = draw;
    }

    delete[] temp;
}

void drawWithoutReplacementSimple(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator, size_t max, const size_t* skip, size_t skip_size, size_t num_samples) {
    result = new size_t[num_samples];
    result_size = num_samples;

    // Set all to not selected
    bool* temp = new bool[max];
    std::fill(temp, temp + max, false);

    std::uniform_int_distribution<size_t> unif_dist(0, max - 1 - skip_size);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t draw;
        do {
            draw = unif_dist(random_number_generator);
            for (size_t j = 0; j < skip_size; ++j) {
                if (draw >= skip[j]) {
                    ++draw;
                }
            }
        } while (temp[draw]);
        temp[draw] = true;
        result[i] = draw;
    }

    delete[] temp;
}

} // namespace ranger


////////////////////////////////////////////////////////////////////////////////////////
void drawWithoutReplacementFisherYates(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator,
    size_t max, size_t num_samples) {

    // Create indices
    result_size = max;
    result = new size_t[max];
    std::iota(result, result + max, 0);

    // Draw without replacement using Fisher Yates algorithm
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t j = i + distribution(random_number_generator) * (max - i);
        std::swap(result[i], result[j]);
    }

    result_size = num_samples;
    // Resize result array to num_samples
    size_t* temp_result = new size_t[num_samples];
    std::copy(result, result + num_samples, temp_result);
    delete[] result;
    result = temp_result;
}

void drawWithoutReplacementFisherYates(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator,
    size_t max, const size_t* skip, size_t skip_size, size_t num_samples) {

    // Create indices
    result_size = max;
    result = new size_t[max];
    std::iota(result, result + max, 0);

    // Skip indices
    for (size_t i = 0; i < skip_size; ++i) {
        size_t skip_index = skip[skip_size - 1 - i];
        for (size_t j = skip_index; j < max - 1; ++j) {
            result[j] = result[j + 1];
        }
        --result_size;
    }

    // Draw without replacement using Fisher Yates algorithm
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t j = i + distribution(random_number_generator) * (result_size - i);
        std::swap(result[i], result[j]);
    }

    // Resize result array to num_samples
    size_t* temp_result = new size_t[num_samples];
    std::copy(result, result + num_samples, temp_result);
    delete[] result;
    result = temp_result;
}

void drawWithoutReplacementWeighted(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator,
    size_t max_index, size_t num_samples, const double* weights, size_t weights_size) {

    result_size = num_samples;
    result = new size_t[num_samples];

    // Set all to not selected
    bool* temp = new bool[max_index + 1]();
    
    std::discrete_distribution<> weighted_dist(weights, weights + weights_size);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t draw;
        do {
            draw = weighted_dist(random_number_generator);
        } while (temp[draw]);
        temp[draw] = true;
        result[i] = draw;
    }

    delete[] temp;
}

double mostFrequentValue(const std::unordered_map<double, size_t>& class_count,
    std::mt19937_64 random_number_generator) {

    std::vector<double> major_classes;
    size_t max_count = 0;

    for (auto& class_value : class_count) {
        if (class_value.second > max_count) {
            max_count = class_value.second;
            major_classes.clear();
            major_classes.push_back(class_value.first);
        } else if (class_value.second == max_count) {
            major_classes.push_back(class_value.first);
        }
    }

    if (major_classes.size() == 1) {
        return major_classes[0];
    } else {
        std::uniform_int_distribution<size_t> unif_dist(0, major_classes.size() - 1);
        return major_classes[unif_dist(random_number_generator)];
    }
}

double computeConcordanceIndex(const Data& data, const double* sum_chf, size_t sum_chf_size,
    const size_t* sample_IDs, size_t sample_IDs_size, double* prediction_error_casewise) {

    double concordance = 0;
    double permissible = 0;

    double* concordance_casewise = nullptr;
    double* permissible_casewise = nullptr;
    if (prediction_error_casewise) {
        concordance_casewise = new double[sum_chf_size]();
        permissible_casewise = new double[sum_chf_size]();
    }

    for (size_t i = 0; i < sum_chf_size; ++i) {
        size_t sample_i = i;
        if (sample_IDs_size > 0) {
            sample_i = sample_IDs[i];
        }
        double time_i = data.get_y(sample_i, 0);
        double status_i = data.get_y(sample_i, 1);

        double conc = (prediction_error_casewise ? concordance_casewise[i] : 0);
        double perm = (prediction_error_casewise ? permissible_casewise[i] : 0);

        for (size_t j = i + 1; j < sum_chf_size; ++j) {
            size_t sample_j = j;
            if (sample_IDs_size > 0) {
                sample_j = sample_IDs[j];
            }
            double time_j = data.get_y(sample_j, 0);
            double status_j = data.get_y(sample_j, 1);

            if (time_i < time_j && status_i == 0) continue;
            if (time_j < time_i && status_j == 0) continue;
            if (time_i == time_j && status_i == status_j) continue;

            double co;
            if (time_i < time_j && sum_chf[i] > sum_chf[j]) co = 1;
            else if (time_j < time_i && sum_chf[j] > sum_chf[i]) co = 1;
            else if (sum_chf[i] == sum_chf[j]) co = 0.5;
            else co = 0;

            conc += co;
            perm += 1;

            if (prediction_error_casewise) {
                concordance_casewise[j] += co;
                permissible_casewise[j] += 1;
            }
        }

        concordance += conc;
        permissible += perm;
        if (prediction_error_casewise) {
            concordance_casewise[i] = conc;
            permissible_casewise[i] = perm;
        }
    }

    if (prediction_error_casewise) {
        for (size_t i = 0; i < sum_chf_size; ++i) {
            prediction_error_casewise[i] = 1 - concordance_casewise[i] / permissible_casewise[i];
        }
        delete[] concordance_casewise;
        delete[] permissible_casewise;
    }

    return (concordance / permissible);
}

std::string uintToString(uint number) {
#if WIN_R_BUILD == 1
    std::stringstream temp;
    temp << number;
    return temp.str();
#else
    return std::to_string(number);
#endif
}

std::string beautifyTime(uint seconds) {
    std::string result;

    uint out_seconds = seconds % 60;
    result = uintToString(out_seconds) + " seconds";
    uint out_minutes = (seconds / 60) % 60;
    if (seconds / 60 == 0) {
        return result;
    } else if (out_minutes == 1) {
        result = "1 minute, " + result;
    } else {
        result = uintToString(out_minutes) + " minutes, " + result;
    }
    uint out_hours = (seconds / 3600) % 24;
    if (seconds / 3600 == 0) {
        return result;
    } else if (out_hours == 1) {
        result = "1 hour, " + result;
    } else {
        result = uintToString(out_hours) + " hours, " + result;
    }
    uint out_days = seconds / 86400;
    if (out_days == 0) {
        return result;
    } else if (out_days == 1) {
        result = "1 day, " + result;
    } else {
        result = uintToString(out_days) + " days, " + result;
    }
    return result;
}
///////////////////////////////////////////////////////////////////////////////////////////

void drawWithoutReplacementFisherYates(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator,
    size_t max, size_t num_samples) {

    // Create indices
    result_size = max;
    result = new size_t[max];
    std::iota(result, result + max, 0);

    // Draw without replacement using Fisher Yates algorithm
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t j = i + distribution(random_number_generator) * (max - i);
        std::swap(result[i], result[j]);
    }

    result_size = num_samples;
    // Resize result array to num_samples
    size_t* temp_result = new size_t[num_samples];
    std::copy(result, result + num_samples, temp_result);
    delete[] result;
    result = temp_result;
}

void drawWithoutReplacementFisherYates(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator,
    size_t max, const size_t* skip, size_t skip_size, size_t num_samples) {

    // Create indices
    result_size = max;
    result = new size_t[max];
    std::iota(result, result + max, 0);

    // Skip indices
    for (size_t i = 0; i < skip_size; ++i) {
        size_t skip_index = skip[skip_size - 1 - i];
        for (size_t j = skip_index; j < result_size - 1; ++j) {
            result[j] = result[j + 1];
        }
        --result_size;
    }

    // Draw without replacement using Fisher Yates algorithm
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t j = i + distribution(random_number_generator) * (result_size - i);
        std::swap(result[i], result[j]);
    }

    // Resize result array to num_samples
    size_t* temp_result = new size_t[num_samples];
    std::copy(result, result + num_samples, temp_result);
    delete[] result;
    result = temp_result;
}

void drawWithoutReplacementWeighted(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator,
    size_t max_index, size_t num_samples, const double* weights, size_t weights_size) {

    result_size = num_samples;
    result = new size_t[num_samples];

    // Set all to not selected
    bool* temp = new bool[max_index + 1]();
    
    std::discrete_distribution<> weighted_dist(weights, weights + weights_size);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t draw;
        do {
            draw = weighted_dist(random_number_generator);
        } while (temp[draw]);
        temp[draw] = true;
        result[i] = draw;
    }

    delete[] temp;
}

double mostFrequentValue(const std::unordered_map<double, size_t>& class_count,
    std::mt19937_64 random_number_generator) {

    double* major_classes = nullptr;
    size_t major_classes_size = 0;
    size_t max_count = 0;

    for (const auto& class_value : class_count) {
        if (class_value.second > max_count) {
            max_count = class_value.second;
            delete[] major_classes;
            major_classes = new double[1];
            major_classes_size = 1;
            major_classes[0] = class_value.first;
        } else if (class_value.second == max_count) {
            double* temp_major_classes = new double[major_classes_size + 1];
            std::copy(major_classes, major_classes + major_classes_size, temp_major_classes);
            delete[] major_classes;
            major_classes = temp_major_classes;
            major_classes[major_classes_size] = class_value.first;
            ++major_classes_size;
        }
    }

    double result;
    if (major_classes_size == 1) {
        result = major_classes[0];
    } else {
        std::uniform_int_distribution<size_t> unif_dist(0, major_classes_size - 1);
        result = major_classes[unif_dist(random_number_generator)];
    }

    delete[] major_classes;
    return result;
}

double computeConcordanceIndex(const Data& data, const double* sum_chf, size_t sum_chf_size,
    const size_t* sample_IDs, size_t sample_IDs_size, double* prediction_error_casewise) {

    double concordance = 0;
    double permissible = 0;

    double* concordance_casewise = nullptr;
    double* permissible_casewise = nullptr;
    if (prediction_error_casewise) {
        concordance_casewise = new double[sum_chf_size]();
        permissible_casewise = new double[sum_chf_size]();
    }

    for (size_t i = 0; i < sum_chf_size; ++i) {
        size_t sample_i = i;
        if (sample_IDs_size > 0) {
            sample_i = sample_IDs[i];
        }
        double time_i = data.get_y(sample_i, 0);
        double status_i = data.get_y(sample_i, 1);

        double conc = (prediction_error_casewise ? concordance_casewise[i] : 0);
        double perm = (prediction_error_casewise ? permissible_casewise[i] : 0);

        for (size_t j = i + 1; j < sum_chf_size; ++j) {
            size_t sample_j = j;
            if (sample_IDs_size > 0) {
                sample_j = sample_IDs[j];
            }
            double time_j = data.get_y(sample_j, 0);
            double status_j = data.get_y(sample_j, 1);

            if (time_i < time_j && status_i == 0) continue;
            if (time_j < time_i && status_j == 0) continue;
            if (time_i == time_j && status_i == status_j) continue;

            double co;
            if (time_i < time_j && sum_chf[i] > sum_chf[j]) co = 1;
            else if (time_j < time_i && sum_chf[j] > sum_chf[i]) co = 1;
            else if (sum_chf[i] == sum_chf[j]) co = 0.5;
            else co = 0;

            conc += co;
            perm += 1;

            if (prediction_error_casewise) {
                concordance_casewise[j] += co;
                permissible_casewise[j] += 1;
            }
        }

        concordance += conc;
        permissible += perm;
        if (prediction_error_casewise) {
            concordance_casewise[i] = conc;
            permissible_casewise[i] = perm;
        }
    }

    if (prediction_error_casewise) {
        for (size_t i = 0; i < sum_chf_size; ++i) {
            prediction_error_casewise[i] = 1 - concordance_casewise[i] / permissible_casewise[i];
        }
        delete[] concordance_casewise;
        delete[] permissible_casewise;
    }

    return (concordance / permissible);
}

std::string uintToString(uint number) {
#if WIN_R_BUILD == 1
    std::stringstream temp;
    temp << number;
    return temp.str();
#else
    return std::to_string(number);
#endif
}

std::string beautifyTime(uint seconds) { // #nocov start
    std::string result;

    // Add seconds, minutes, hours, days if larger than zero
    uint out_seconds = seconds % 60;
    result = uintToString(out_seconds) + " seconds";
    uint out_minutes = (seconds / 60) % 60;
    if (seconds / 60 == 0) {
        return result;
    } else if (out_minutes == 1) {
        result = "1 minute, " + result;
    } else {
        result = uintToString(out_minutes) + " minutes, " + result;
    }
    uint out_hours = (seconds / 3600) % 24;
    if (seconds / 3600 == 0) {
        return result;
    } else if (out_hours == 1) {
        result = "1 hour, " + result;
    } else {
        result = uintToString(out_hours) + " hours, " + result;
    }
    uint out_days = seconds / 86400;
    if (out_days == 0) {
        return result;
    } else if (out_days == 1) {
        result = "1 day, " + result;
    } else {
        result = uintToString(out_days) + " days, " + result;
    }
    return result;
} // #nocov end

////////////////////////////////////////////////////////////////////////////////////////


// #nocov start
size_t roundToNextMultiple(size_t value, uint multiple) {
    if (multiple == 0) {
        return value;
    }

    size_t remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }

    return value + multiple - remainder;
}

void splitString(std::string** result, size_t* result_size, const std::string& input, char split_char) {
    std::istringstream ss(input);
    std::string token;
    size_t count = 0;

    // Initial allocation (estimate size)
    size_t initial_size = 10;
    *result = new std::string[initial_size];
    
    while (std::getline(ss, token, split_char)) {
        if (count >= initial_size) {
            // Resize the array
            size_t new_size = initial_size * 2;
            std::string* new_array = new std::string[new_size];
            std::copy(*result, *result + initial_size, new_array);
            delete[] *result;
            *result = new_array;
            initial_size = new_size;
        }
        (*result)[count++] = token;
    }
    *result_size = count;
}

void splitString(double** result, size_t* result_size, const std::string& input, char split_char) {
    std::istringstream ss(input);
    std::string token;
    size_t count = 0;

    // Initial allocation (estimate size)
    size_t initial_size = 10;
    *result = new double[initial_size];
    
    while (std::getline(ss, token, split_char)) {
        if (count >= initial_size) {
            // Resize the array
            size_t new_size = initial_size * 2;
            double* new_array = new double[new_size];
            std::copy(*result, *result + initial_size, new_array);
            delete[] *result;
            *result = new_array;
            initial_size = new_size;
        }
        (*result)[count++] = std::stod(token);
    }
    *result_size = count;
}

void shuffleAndSplit(size_t** first_part, size_t* first_part_size, size_t** second_part, size_t* second_part_size, size_t n_all, size_t n_first, std::mt19937_64 random_number_generator) {
    // Allocate and fill with 0..n_all-1
    *first_part = new size_t[n_all];
    std::iota(*first_part, *first_part + n_all, 0);

    // Shuffle
    std::shuffle(*first_part, *first_part + n_all, random_number_generator);

    // Copy to second part
    *second_part = new size_t[n_all - n_first];
    std::copy(*first_part + n_first, *first_part + n_all, *second_part);

    // Resize first part
    *first_part_size = n_first;
    size_t* temp_first_part = new size_t[n_first];
    std::copy(*first_part, *first_part + n_first, temp_first_part);
    delete[] *first_part;
    *first_part = temp_first_part;
}

void shuffleAndSplitAppend(size_t** first_part, size_t* first_part_size, size_t** second_part, size_t* second_part_size, size_t n_all, size_t n_first, const size_t* mapping, size_t mapping_size, std::mt19937_64 random_number_generator) {
    // Old end is start position for new data
    size_t first_old_size = *first_part_size;
    size_t second_old_size = *second_part_size;

    // Reserve space
    size_t* temp_first_part = new size_t[first_old_size + n_all];
    std::copy(*first_part, *first_part + first_old_size, temp_first_part);
    delete[] *first_part;
    *first_part = temp_first_part;

    // Fill with 0..n_all-1 and shuffle
    std::iota(*first_part + first_old_size, *first_part + first_old_size + n_all, 0);
    std::shuffle(*first_part + first_old_size, *first_part + first_old_size + n_all, random_number_generator);

    // Mapping
    for (size_t i = first_old_size; i < first_old_size + n_all; ++i) {
        (*first_part)[i] = mapping[(*first_part)[i]];
    }

    // Copy to second part
    *second_part = new size_t[second_old_size + n_all - n_first];
    std::copy(*first_part + first_old_size + n_first, *first_part + first_old_size + n_all, *second_part + second_old_size);

    // Resize first part
    *first_part_size = first_old_size + n_first;
    size_t* temp_first_part_new = new size_t[*first_part_size];
    std::copy(*first_part, *first_part + *first_part_size, temp_first_part_new);
    delete[] *first_part;
    *first_part = temp_first_part_new;
}

std::string checkUnorderedVariables(const Data& data, const std::string* unordered_variable_names, size_t unordered_variable_names_size) {
    size_t num_rows = data.getNumRows();
    size_t* sampleIDs = new size_t[num_rows];
    std::iota(sampleIDs, sampleIDs + num_rows, 0);

    // Check for all unordered variables
    for (size_t i = 0; i < unordered_variable_names_size; ++i) {
        std::string variable_name = unordered_variable_names[i];
        size_t varID = data.getVariableID(variable_name);
        double* all_values;
        size_t all_values_size;
        data.getAllValues(&all_values, &all_values_size, sampleIDs, varID, 0, num_rows);

        // Check level count
        size_t max_level_count = 8 * sizeof(size_t) - 1;
        if (all_values_size > max_level_count) {
            delete[] sampleIDs;
            delete[] all_values;
            return "Too many levels in unordered categorical variable " + variable_name + ". Only " + uintToString(max_level_count) + " levels allowed on this system.";
        }

        // Check positive integers
        if (!checkPositiveIntegers(all_values, all_values_size)) {
            delete[] sampleIDs;
            delete[] all_values;
            return "Not all values in unordered categorical variable " + variable_name + " are positive integers.";
        }
        delete[] all_values;
    }
    delete[] sampleIDs;
    return "";
}

bool checkPositiveIntegers(const double* all_values, size_t all_values_size) {
    for (size_t i = 0; i < all_values_size; ++i) {
        if (all_values[i] < 1 || !(std::floor(all_values[i]) == all_values[i])) {
            return false;
        }
    }
    return true;
}

double maxstatPValueLau92(double b, double minprop, double maxprop) {
    if (b < 1) {
        return 1.0;
    }

    // Compute only once (minprop/maxprop don't change during runtime)
    static double logprop = std::log((maxprop * (1 - minprop)) / ((1 - maxprop) * minprop));

    double db = dstdnorm(b);
    double p = 4 * db / b + db * (b - 1 / b) * logprop;

    return (p > 0) ? p : 0;
}

////////////////////////////////////////////////////////////////////////////////////////
// Function prototypes
double maxstatPValueLau94(double b, double minprop, double maxprop, size_t N, const size_t* m, size_t m_size);
double maxstatPValueUnadjusted(double b);
double dstdnorm(double x);
double pstdnorm(double x);
double* adjustPvalues(const double* unadjusted_pvalues, size_t num_pvalues);
double* logrankScores(const double* time, const double* status, size_t n);
void maxstat(const double* scores, const double* x, const size_t* indices, size_t n, double& best_maxstat, double& best_split_value, double minprop, double maxprop);
size_t* numSamplesLeftOfCutpoint(const double* x, const size_t* indices, size_t n);

// Function implementations
double maxstatPValueLau94(double b, double minprop, double maxprop, size_t N, const size_t* m, size_t m_size) {
    double D = 0;
    for (size_t i = 0; i < m_size - 1; ++i) {
        double m1 = static_cast<double>(m[i]);
        double m2 = static_cast<double>(m[i + 1]);

        double t = sqrt(1.0 - m1 * (N - m2) / ((N - m1) * m2));
        D += 1 / M_PI * exp(-b * b / 2) * (t - (b * b / 4 - 1) * (t * t * t) / 6);
    }

    return 2 * (1 - pstdnorm(b)) + D;
}

double maxstatPValueUnadjusted(double b) {
    return 2 * pstdnorm(-b);
}

double dstdnorm(double x) {
    return exp(-0.5 * x * x) / sqrt(2 * M_PI);
}

double pstdnorm(double x) {
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}

double* adjustPvalues(const double* unadjusted_pvalues, size_t num_pvalues) {
    double* adjusted_pvalues = new double[num_pvalues];
    std::memset(adjusted_pvalues, 0, num_pvalues * sizeof(double));

    // Get order of p-values
    size_t* indices = order(unadjusted_pvalues, num_pvalues, true);

    // Compute adjusted p-values
    adjusted_pvalues[indices[0]] = unadjusted_pvalues[indices[0]];
    for (size_t i = 1; i < num_pvalues; ++i) {
        size_t idx = indices[i];
        size_t idx_last = indices[i - 1];

        adjusted_pvalues[idx] = std::min(adjusted_pvalues[idx_last],
            (double) num_pvalues / (double) (num_pvalues - i) * unadjusted_pvalues[idx]);
    }

    delete[] indices;
    return adjusted_pvalues;
}

double* logrankScores(const double* time, const double* status, size_t n) {
    double* scores = new double[n];
    std::memset(scores, 0, n * sizeof(double));

    // Get order of timepoints
    size_t* indices = order(time, n, false);

    // Compute scores
    double cumsum = 0;
    size_t last_unique = static_cast<size_t>(-1);
    for (size_t i = 0; i < n; ++i) {
        // Continue if next value is the same
        if (i < n - 1 && time[indices[i]] == time[indices[i + 1]]) {
            continue;
        }

        // Compute sum and scores for all non-unique values in a row
        for (size_t j = last_unique + 1; j <= i; ++j) {
            cumsum += status[indices[j]] / (n - i);
        }
        for (size_t j = last_unique + 1; j <= i; ++j) {
            scores[indices[j]] = status[indices[j]] - cumsum;
        }

        // Save last computed value
        last_unique = i;
    }

    delete[] indices;
    return scores;
}

void maxstat(const double* scores, const double* x, const size_t* indices, size_t n, double& best_maxstat, double& best_split_value, double minprop, double maxprop) {
    double sum_all_scores = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_all_scores += scores[indices[i]];
    }

    // Compute sum of differences from mean for variance
    double mean_scores = sum_all_scores / n;
    double sum_mean_diff = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_mean_diff += (scores[i] - mean_scores) * (scores[i] - mean_scores);
    }

    // Get smallest and largest split to consider, -1 for compatibility with R maxstat
    size_t minsplit = 0;
    if (n * minprop > 1) {
        minsplit = n * minprop - 1;
    }
    size_t maxsplit = n * maxprop - 1;

    // For all unique x-values
    best_maxstat = -1;
    best_split_value = -1;
    double sum_scores = 0;
    size_t n_left = 0;
    for (size_t i = 0; i <= maxsplit; ++i) {
        sum_scores += scores[indices[i]];
        n_left++;

        // Don't consider splits smaller than minsplit for splitting (but count)
        if (i < minsplit) {
            continue;
        }

        // Consider only unique values
        if (i < n - 1 && x[indices[i]] == x[indices[i + 1]]) {
            continue;
        }

        // If value is largest possible value, stop
        if (x[indices[i]] == x[indices[n - 1]]) {
            break;
        }

        double S = sum_scores;
        double E = (double) n_left / (double) n * sum_all_scores;
        double V = (double) n_left * (double) (n - n_left) / (double) (n * (n - 1)) * sum_mean_diff;
        double T = fabs((S - E) / sqrt(V));

        if (T > best_maxstat) {
            best_maxstat = T;

            // Use mid-point split if possible
            if (i < n - 1) {
                best_split_value = (x[indices[i]] + x[indices[i + 1]]) / 2;
            } else {
                best_split_value = x[indices[i]];
            }
        }
    }
}

size_t* numSamplesLeftOfCutpoint(const double* x, const size_t* indices, size_t n) {
    size_t* num_samples_left = new size_t[n];
    num_samples_left[0] = 1;

    for (size_t i = 1; i < n; ++i) {
        if (x[indices[i]] == x[indices[i - 1]]) {
            num_samples_left[i] = num_samples_left[i - 1] + 1;
        } else {
            num_samples_left[i] = num_samples_left[i - 1] + 1;
        }
    }

    return num_samples_left;
}

// Dummy order function (you'll need to implement or provide the actual function)
size_t* order(const double* values, size_t size, bool ascending) {
    size_t* indices = new size_t[size];
    std::iota(indices, indices + size, 0);

    std::sort(indices, indices + size, [values, ascending](size_t i1, size_t i2) {
        return ascending ? values[i1] < values[i2] : values[i1] > values[i2];
    });

    return indices;
}

// Implement the rest of the missing functions and include necessary headers and utilities


////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <sstream>


// Function prototypes
std::stringstream& readFromStream(std::stringstream& in, double* token);
double betaLogLik(double y, double mean, double phi);

// Function implementations
std::stringstream& readFromStream(std::stringstream& in, double* token) {
    if (!(in >> *token) && (std::fpclassify(*token) == FP_SUBNORMAL)) {
        in.clear();
    }
    return in;
}

double betaLogLik(double y, double mean, double phi) {
    // Avoid 0 and 1
    if (y < std::numeric_limits<double>::epsilon()) {
        y = std::numeric_limits<double>::epsilon();
    } else if (y >= 1) {
        y = 1 - std::numeric_limits<double>::epsilon();
    }
    if (mean < std::numeric_limits<double>::epsilon()) {
        mean = std::numeric_limits<double>::epsilon();
    } else if (mean >= 1) {
        mean = 1 - std::numeric_limits<double>::epsilon();
    }
    if (phi < std::numeric_limits<double>::epsilon()) {
        phi = std::numeric_limits<double>::epsilon();
    } else if (phi >= 1) {
        phi = 1 - std::numeric_limits<double>::epsilon();
    }

    return (lgamma(phi) - lgamma(mean * phi) - lgamma((1 - mean) * phi) 
            + (mean * phi - 1) * log(y)
            + ((1 - mean) * phi - 1) * log(1 - y));
}

// Example usage
int main() {
    std::stringstream ss("0.5");
    double token;
    readFromStream(ss, &token);
    std::cout << "Token: " << token << std::endl;

    double result = betaLogLik(0.5, 0.5, 1.0);
    std::cout << "Beta Log Likelihood: " << result << std::endl;

    return 0;
}

