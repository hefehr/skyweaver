#include "utils.hpp"
#include <glob.h>
#include <memory.h>

std::vector<std::string> globFiles(const std::string& pattern) {
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // Perform glob operation
    int result = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (result != 0) {
        globfree(&glob_result);
        std::cerr << "Failed to glob files: ";
        switch(result) {
            case GLOB_NOSPACE:
                std::cerr << "Out of memory\n";
                break;
            case GLOB_ABORTED:
                std::cerr << "Read error\n";
                break;
            case GLOB_NOMATCH:
                std::cerr << "No matches found\n";
                break;
            default:
                std::cerr << "Unknown error\n";
                break;
        }
        return {};
    }

    std::vector<std::string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(std::string(glob_result.gl_pathv[i]));
    }

    globfree(&glob_result);
    return filenames;
}


