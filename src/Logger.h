/**
 *  @file Logger.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef LOGGER_H_
#define LOGGER_H_
#include <ctime>
#include <stdarg.h>
#include <stdio.h>
#include <string>

namespace mass {

/**
 * This class is intended to log messages using printf style string formatting.
 */
class Logger {

public:

	/**
	 * Closes the logger. This is called by default in the destructor.
	 */
	static void close();

	/**
	 * Log a formatted message with the flag [DEBUG]
	 * @param fmt the format string
	 * @param va_arg the arguments for the formatted string
	 */
	static void debug(char *fmt, ...);

	/**
	 * Log a formatted message with the flag [WARNING]
	 * @param fmt the format string
	 * @param va_arg the arguments for the formatted string
	 */
	static void warn(char *fmt, ...);

	/**
	 * Log a formatted message with the flag [ERROR]
	 * @param fmt the format string
	 * @param va_arg the arguments for the formatted string
	 */
	static void error(char *fmt, ...);

	/**
	 * Log a formatted message with the flag [INFO]
	 * @param fmt the format string
	 * @param va_arg the arguments for the formatted string
	 */
	static void info(char *fmt, ...);

	/**
	 * Sets the file to be used for logging. Closes out any existing logger if
	 * one already exists. If filename does not exist, it will be created.
	 *
	 * @param filename the relative file path to where the log file is to be
	 * stored or created.
	 */
	static void setLogFile(std::string filename);

private:

	static Logger instance;
	static FILE *pFile; /**< The output file.*/
	static bool isOpen; /**< Tracks if an output file is open for this logger.*/

	static struct std::tm * ptm; /**< A time struct used for time stamping log events.*/
	static std::time_t rawtime; /**< The number of seconds since epoch.*/

	const static size_t BUFSIZE = 80; /**< The max chars for a time stamp.*/
	static char buf[BUFSIZE]; /**< The space for the most recent time stamp.*/

	/**
	 * Stores the local time in buf
	 * @return buf
	 */
	static char *getLocalTime();
};

} /* namespace mass */
#endif /* LOGGER_H_ */
