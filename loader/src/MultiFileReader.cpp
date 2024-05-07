#include "MultiFileReader.hpp"



MultiFileReader::MultiFileReader(const std::vector<std::string>& fileNames, long headerSize, bool checkContinuity) : 
    files(fileNames), currentFileIndex(0), currentPosition(0), eofFlag(false), headerSize(headerSize), checkContinuity(checkContinuity) {
    std::size_t totalSize = 0;
    for (const auto& file : files) {
        std::ifstream f(file, std::ifstream::ate | std::ifstream::binary);
        f.seekg(0, std::ios::end); // Move to the end of the file
        std::size_t size = f.tellg();
        totalSize += (size - headerSize); // skip header
        sizes.push_back(size - headerSize);
        f.close();
    }
    this->totalSize = totalSize;
    if (!files.empty()) {
        read_header();  // Read header from the first file to set headerSize
    }
}

void MultiFileReader::read_header() {
    std::ifstream f(files[0], std::ifstream::binary);
    std::vector<char> headerBytes(headerSize);
    char headerBytes[headerSize];
    f.read(headerBytes.data(), headerSize);
    this->header = std::make_unique<psrdada_cpp::PsrDadaHeader>(headerBytes);
    f.close();
}


void MultiFileReader::open() {
    std::lock_guard<std::mutex> lock(mtx);     /* mutex released automatically at end of scope*/
    currentStream.open(files[currentFileIndex], std::ifstream::binary);
    if (headerSize > 0 && currentStream.good()) {
        currentStream.seekg(headerSize, std::ios::beg);  // Skip header for subsequent files
    }
    currentPosition = headerSize;
    eofFlag = false; 
}

void MultiFileReader::open_next() {
    std::lock_guard<std::mutex> lock(mtx);
    currentFileIndex++;
    if (currentFileIndex < files.size()) {
        currentStream.close();
        open();
    } else {
        eofFlag = true;
    }
}

void MultiFileReader::open_previous() {
    std::lock_guard<std::mutex> lock(mtx);
    currentFileIndex--;
    if (currentFileIndex >= 0) {
        currentStream.close();
        open();
    } else {
        eofFlag = true;
    }
}


void MultiFileReader::seekg(long pos, std::ios_base::seekdir dir) {
    std::lock_guard<std::mutex> lock(mtx);
    long targetPos = (dir == std::ios_base::cur) ? currentPosition + pos : pos;
    if (dir == std::ios_base::end && pos < 0) targetPos = this->totalSize + pos;

    long cumulativeSize = 0;
    for (size_t i = 0; i < sizes.size(); i++) {
        cumulativeSize += sizes[i];
        if (targetPos < cumulativeSize) {
            long filePos = targetPos - (cumulativeSize - sizes[i]); // position in file
            if (currentFileIndex != i) {
                currentStream.close();
                currentFileIndex = i;
                currentStream.open(files[currentFileIndex], std::ifstream::binary);
                }
                
            currentStream.seekg(headerSize + filePos, std::ios::beg);
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

std::streamsize MultiFileReader::read(std::unique_ptr<std::vector<char>> buffer, std::streamsize bytes) {
    std::lock_guard<std::mutex> lock(mtx);
    std::streamsize totalRead = 0;
    while (bytes > 0) {
        currentStream.read(buffer->data() + totalRead, bytes);
        std::streamsize bytesRead = currentStream.gcount();
        if (bytesRead == 0 || bytesRead < bytes) {  // End of file or error
            open_next();
        } else {
            totalRead += bytesRead;
            bytes -= bytesRead;
            currentPosition += bytesRead;
        }

        if(eofFlag) break;  
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
