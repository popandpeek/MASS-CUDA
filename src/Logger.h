/**
 *  @file Logger.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef LOGGER_H_
#define LOGGER_H_
#include <stdarg.h>
#include <stdio.h>
#include <string>

namespace mass {

/**
 * This class is intended to log messages using printf style string formatting.
 */
class Logger {

public:
	Logger();
	virtual ~Logger();


	/**
	 * Log a formatted message with the flag [DEBUG]
	 * @param fmt the format string
	 * @param va_arg the arguments for the formatted string
	 */
	void debug(char *fmt, ...);

	/**
	 * Log a formatted message with the flag [WARNING]
	 * @param fmt the format string
	 * @param va_arg the arguments for the formatted string
	 */
	void warn(char *fmt, ...);

	/**
	 * Log a formatted message with the flag [ERROR]
	 * @param fmt the format string
	 * @param va_arg the arguments for the formatted string
	 */
	void error(char *fmt, ...);

	/**
	 * Log a formatted message with the flag [INFO]
	 * @param fmt the format string
	 * @param va_arg the arguments for the formatted string
	 */
	void info(char *fmt, ...);

	/**
	 * Sets the file to be used for logging. Closes out any existing logger if
	 * one already exists. If filename does not exist, it will be created.
	 *
	 * @param filename the relative file path to where the log file is to be
	 * stored or created.
	 */
	void setLogFile(std::string filename);

private:
	FILE *pFile; /**< The output file.*/
	bool isOpen; /**< Tracks if an output file is open for this logger.*/

	struct std::tm * ptm; /**< A time struct used for time stamping log events.*/
	std::time_t rawtime; /**< The number of seconds since epoch.*/

	const static size_t BUFSIZE = 80; /**< The max chars for a time stamp.*/
	char buf[BUFSIZE]; /**< The space for the most recent time stamp.*/

	/**
	 * Stores the local time in buf
	 * @return buf
	 */
	char *getLocalTime();

	/**
	 * Closes the logger. This is called by default in the destructor.
	 */
	void close();

};

} /* namespace mass */
#endif /* LOGGER_H_ */
