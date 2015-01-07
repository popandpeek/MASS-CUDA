/**
 * Logger.cpp
 *
 *  Author: Nate Hart
 *  Created on: Oct 21, 2014
 */

#include "Logger.h"
#define DEBUG

namespace mass {

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

void Logger::info(char* fmt, ...) {
	if (!isOpen) {
		setLogFile("mass_log.txt");
	}

	fprintf(pFile, "%s [INFO]    ", Logger::getLocalTime());
	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt, args);
	fprintf(pFile, "\n");
	va_end(args);
	fflush(pFile); // make sure all logs make it out in event of a crash
}

void Logger::warn(char* fmt, ...) {
	if (!isOpen) {
		setLogFile("mass_log.txt");
	}

	fprintf(pFile, "%s [WARNING] ", Logger::getLocalTime());
	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt, args);
	fprintf(pFile, "\n");
	va_end(args);
	fflush(pFile); // make sure all logs make it out in event of a crash
}

void Logger::error(char* fmt, ...) {
	if (!isOpen) {
		setLogFile("mass_log.txt");
	}

	fprintf(pFile, "%s [ERROR]   ", Logger::getLocalTime());
	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt, args);
	fprintf(pFile, "\n");
	va_end(args);
	fflush(pFile); // make sure all logs make it out in event of a crash
}

void Logger::debug(char* fmt, ...) {
#ifdef DEBUG // gives ability to turn off this logging functionality from a single place
	if (!isOpen) {
		setLogFile("mass_log.txt");
	}

	fprintf(pFile, "%s [DEBUG]   ", Logger::getLocalTime());
	va_list args;
	va_start(args, fmt);
	vfprintf(pFile, fmt, args);
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
		debug("Log file switched to %s.", filename.c_str());
		fclose(pFile);
	}

	pFile = fopen(filename.c_str(), "a");
	strftime(buf, BUFSIZE, "%a %Y/%m/%d %H:%M:%S ", ptm); // get detailed time
	fprintf(pFile, "%s [INFO] Logger initialized.\n", buf);
	fflush(pFile);
	isOpen = true;
}

} /* namespace mass */
