/*
 * Copyright (c) 2016-2017, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#include "board.hpp"
#include <modm/architecture/interface/delay.hpp>
#include <modm/architecture/interface/assert.hpp>

Board::LoggerDevice loggerDevice;

// Set all four logger streams to use the UART
modm::log::Logger modm::log::debug(loggerDevice);
modm::log::Logger modm::log::info(loggerDevice);
modm::log::Logger modm::log::warning(loggerDevice);
modm::log::Logger modm::log::error(loggerDevice);

// Default all calls to printf to the UART
modm_extern_c void
putchar_(char c)
{
	loggerDevice.write(c);
}

modm_extern_c void
modm_abandon(const modm::AssertionInfo &info)
{
	MODM_LOG_ERROR << "Assertion '" << info.name << "'";
	if (info.context != uintptr_t(-1)) {
		MODM_LOG_ERROR << " @ " << (void *) info.context <<
						 " (" << (uint32_t) info.context << ")";
	}
	#if MODM_ASSERTION_INFO_HAS_DESCRIPTION
	MODM_LOG_ERROR << " failed!\n  " << info.description << "\nAbandoning...\n";
	#else
	MODM_LOG_ERROR << " failed!\nAbandoning...\n";
	#endif
	Board::Leds::setOutput();
	for(int times=10; times>=0; times--)
	{
		Board::Leds::write(1);
		modm::delay_ms(20);
		Board::Leds::write(0);
		modm::delay_ms(180);
	}
	// Do not flush here otherwise you may deadlock due to waiting on the UART
	// interrupt which may never be executed when abandoning in a higher
	// priority Interrupt!!!
	// MODM_LOG_ERROR << modm::flush;
}
