/**
 * Logger.cpp
 *
 *  Author: Nate Hart
 *  Created on: Oct 21, 2014
 */

#include "Logger.h"

namespace mass {

const std::string Logger::DEFAULT_LOG_FILE = "mass_log.txt";

// static initialization
FILE *Logger::pFile = NULL; /**< The output file.*/
bool Logger::isOpen = false; /**< Tracks if an output file is open for this logger.*/
struct std::tm * Logger::ptm; /**< A time struct used for time stamping log events.*/
std::time_t Logger::rawtime; /**< The number of seconds since epoch.*/
char Logger::buf[BUFSIZE]; /**< The space for the most recent time stamp.*/

char *Logger::getLocalTime() {
	// get local time
	time(&rawtime);
	ptm = localtime(&rawtime);
	strftime(buf, BUFSIZE, "%H:%M:%S ", ptm); // record time
	return buf;
}

void Logger::close() {
	if (isOpen) {
		fclose(pFile);
		isOpen = false;
	}
}

void Logger::info(std::string fmt, ...) {
	if (!isOpen) {
		setLogFile(DEFAULT_LOG_FILE);
	}

	fprintf(pFile, "%s [INFO]    ", Logger::getLocalTime());
	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt.c_str(), args);
	fprintf(pFile, "\n");
	va_end(args);
	fflush(pFile); // make sure all logs make it out in event of a crash
}

void Logger::print(std::string fmt, ...) {
	if (!isOpen) {
		setLogFile(DEFAULT_LOG_FILE);
	}

	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt.c_str(), args);
	va_end(args);
	fflush(pFile); // make sure all logs make it out in event of a crash
}

void Logger::warn(std::string fmt, ...) {
	if (!isOpen) {
		setLogFile(DEFAULT_LOG_FILE);
	}

	fprintf(pFile, "%s [WARNING] ", Logger::getLocalTime());
	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt.c_str(), args);
	fprintf(pFile, "\n");
	va_end(args);
	fflush(pFile); // make sure all logs make it out in event of a crash
}

void Logger::error(std::string fmt, ...) {
	if (!isOpen) {
		setLogFile(DEFAULT_LOG_FILE);
	}

	fprintf(pFile, "%s [ERROR]   ", Logger::getLocalTime());
	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt.c_str(), args);
	fprintf(pFile, "\n");
	va_end(args);
	fflush(pFile); // make sure all logs make it out in event of a crash
}

void Logger::debug(std::string fmt, ...) {
#ifdef DEBUG // gives ability to turn off this logging functionality from a single place
	if (!isOpen) {
		setLogFile(DEFAULT_LOG_FILE);
	}

	fprintf(pFile, "%s [DEBUG]   ", Logger::getLocalTime());
	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt.c_str(), args);
	fprintf(pFile, "\n");
	va_end(args);
	fflush(pFile); // make sure all logs make it out in event of a crash
#endif
}

void Logger::setLogFile(std::string filename) {
	// get local time
	time(&rawtime);
	ptm = localtime(&rawtime);

	if (isOpen) {
		info("Log file switched to %s.", filename.c_str());
		fclose(pFile);
	}

	pFile = fopen(filename.c_str(), "a");
	strftime(buf, BUFSIZE, "%a %Y/%m/%d %H:%M:%S ", ptm); // get detailed time
	fprintf(pFile, "\n\n%s [INFO] Logger initialized.\n", buf);
	fflush(pFile);
	isOpen = true;
}
void Logger::truncateLogfile(std::string filename) {
	close();
	// get local time
	time(&rawtime);
	ptm = localtime(&rawtime);

	pFile = fopen(filename.c_str(), "w");
	strftime(buf, BUFSIZE, "%a %Y/%m/%d %H:%M:%S ", ptm); // get detailed time
	fprintf(pFile, "\n\n%s [INFO] Logger initialized.\n", buf);
	fflush(pFile);
	isOpen = true;
}

} /* namespace mass */
