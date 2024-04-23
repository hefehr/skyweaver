#include "MultiFileReader.hpp"



MultiFileReader::MultiFileReader(const std::vector<std::string>& fileNames, long headerSize) : 
    files(fileNames), currentFileIndex(0), currentPosition(0), eofFlag(false), headerSize(headerSize) {
    for (const auto& file : files) {
        std::ifstream f(file, std::ifstream::ate | std::ifstream::binary);
        sizes.push_back(f.tellg() - headerSize);
        f.close();
    }
    if (!files.empty()) {
        readHeader();  // Read header from the first file to set headerSize
    }
}

void MultiFileReader:read_header() {
    std::ifstream file(files[0], std::ifstream::binary);
    std::vector<char> headerBytes(headerSize);
    char headerBytes[headerSize];
    file.read(headerBytes.data(), headerSize);
    this->header = std::make_unique<psrdada_cpp::PsrDadaHeader>(headerBytes);
    file.close();
}


void MultiFileReader::open() {
    std::lock_guard<std::mutex> lock(mtx);
    currentStream.open(files[currentFileIndex], std::ifstream::binary);
    if (currentFileIndex > 0 && headerSize > 0 && currentStream.good()) {
        currentStream.seekg(headerSize, std::ios::beg);  // Skip header for subsequent files
    }
    currentPosition = 0;
    eofFlag = false;
}


void MultiFileReader::seekg(long pos, std::ios_base::seekdir dir) {
    std::lock_guard<std::mutex> lock(mtx);
    long targetPos = (dir == std::ios_base::cur) ? currentPosition + pos : pos;
    if (dir == std::ios_base::end) {
        long totalSize = 0;
        for (auto size : sizes) totalSize += size;
        targetPos = totalSize + pos;
    }

    long cumulativeSize = 0;
    for (size_t i = 0; i < sizes.size(); i++) {
        cumulativeSize += sizes[i];
        if (targetPos < cumulativeSize) {
            long filePos = targetPos - (cumulativeSize - sizes[i]);
            if (currentFileIndex != i) {
                currentStream.close();
                currentFileIndex = i;
                currentStream.open(files[currentFileIndex], std::ifstream::binary);
                if (headerSize > 0) currentStream.seekg(headerSize + filePos, std::ios::beg);
                else currentStream.seekg(filePos, std::ios::beg);
            } else {
                currentStream.seekg(filePos + headerSize, std::ios::beg);
            }
            currentPosition = targetPos;
            eofFlag = false;
            break;
        }
    }
}

void MultiFileReader::close() {
    std::lock_guard<std::mutex> lock(mtx);
    if (currentStream.is_open()) {
        currentStream.close();
    }
    eofFlag = true;
}

std::streamsize MultiFileReader::read(char* buffer, std::streamsize bytes) {
    std::lock_guard<std::mutex> lock(mtx);
    std::streamsize totalRead = 0;
    while (bytes > 0) {
        if (currentStream.eof()) {
            eofFlag = true;
            break;
        }

        currentStream.read(buffer + totalRead, bytes);
        std::streamsize bytesRead = currentStream.gcount();
        if (bytesRead == 0) {  // End of file or error
            currentFileIndex++;
            if (currentFileIndex < files.size()) {
                currentStream.close();
                open();  // Open next file and continue reading
            } else {
                eofFlag = true;
                break;
            }
        } else {
            totalRead += bytesRead;
            bytes -= bytesRead;
            currentPosition += bytesRead;
        }
    }
    return totalRead;
}

bool MultiFileReader::eof() const {
    return eofFlag;
}

bool MultiFileReader::good() const {
    return currentStream.good() && !eofFlag;
}

long MultiFileReader::tellg() const {
    return currentPosition;
}

template<typename T>
MultiFileReader& operator>>(MultiFileReader& reader, T& value) {
    std::lock_guard<std::mutex> lock(reader.mtx);
    if (!(reader.currentStream >> value)) {
        reader.eofFlag = true;
    }
    return reader;
}
