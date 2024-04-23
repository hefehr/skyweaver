#include "common.hpp"
#include "SkyWeaver.hpp"
#include "loader/MultiFileReader.hpp"

int main(int argc, char** argv) {
    std::vector<std::string> filenames = {"part1.bin", "part2.bin"};
    MultiFileReader reader(filenames, 10); // Skip 10 bytes header in each file
    reader.open();
    const int bufferSize = 1024;
    char buffer[bufferSize];
    std::streamsize bytesRead = reader.read(buffer, bufferSize);

    if (bytesRead > 0) {
        // Process the read binary data
        std::cout << "Read " << bytesRead << " bytes." << std::endl;
    }

    reader.close();
    return 0;

}
