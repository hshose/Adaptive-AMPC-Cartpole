/*
 * Copyright (c) 2009, Martin Rosekeit
 * Copyright (c) 2009-2011, Fabian Greif
 * Copyright (c) 2010-2011, 2013, Georgi Grinshpun
 * Copyright (c) 2013-2014, Sascha Schade
 * Copyright (c) 2013, 2016, Kevin Läufer
 * Copyright (c) 2013-2017, Niklas Hauser
 * Copyright (c) 2018, Lucas Mösch
 * Copyright (c) 2021, Raphael Lehmann
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#include "../device.hpp"
#include "uart_2.hpp"
#include <modm/architecture/interface/atomic_lock.hpp>
#include <modm/architecture/driver/atomic/queue.hpp>

namespace
{
	static modm::atomic::Queue<uint8_t, 2048> txBuffer;
}
namespace modm::platform
{

void
Usart2::writeBlocking(uint8_t data)
{
	while(!UsartHal2::isTransmitRegisterEmpty());
	UsartHal2::write(data);
}

void
Usart2::writeBlocking(const uint8_t *data, std::size_t length)
{
	while (length-- != 0) {
		writeBlocking(*data++);
	}
}

void
Usart2::flushWriteBuffer()
{
	while(!isWriteFinished());
}

bool
Usart2::write(uint8_t data)
{
	if(txBuffer.isEmpty() && UsartHal2::isTransmitRegisterEmpty()) {
		UsartHal2::write(data);
	} else {
		if (!txBuffer.push(data))
			return false;
		// Disable interrupts while enabling the transmit interrupt
		atomic::Lock lock;
		// Transmit Data Register Empty Interrupt Enable
		UsartHal2::enableInterrupt(Interrupt::TxEmpty);
	}
	return true;
}

std::size_t
Usart2::write(const uint8_t *data, std::size_t length)
{
	uint32_t i = 0;
	for (; i < length; ++i)
	{
		if (!write(*data++)) {
			return i;
		}
	}
	return i;
}

bool
Usart2::isWriteFinished()
{
	return txBuffer.isEmpty() && UsartHal2::isTransmitRegisterEmpty();
}

std::size_t
Usart2::transmitBufferSize()
{
	return txBuffer.getSize();
}

std::size_t
Usart2::discardTransmitBuffer()
{
	std::size_t count = 0;
	// disable interrupt since buffer will be cleared
	UsartHal2::disableInterrupt(UsartHal2::Interrupt::TxEmpty);
	while(!txBuffer.isEmpty()) {
		++count;
		txBuffer.pop();
	}
	return count;
}

bool
Usart2::read(uint8_t &data)
{
	if(UsartHal2::isReceiveRegisterNotEmpty()) {
		UsartHal2::read(data);
		return true;
	} else {
		return false;
	}
}

std::size_t
Usart2::read(uint8_t *data, std::size_t length)
{
	(void)length; // avoid compiler warning
	if(read(*data)) {
		return 1;
	} else {
		return 0;
	}
}

std::size_t
Usart2::receiveBufferSize()
{
	return UsartHal2::isReceiveRegisterNotEmpty() ? 1 : 0;
}

std::size_t
Usart2::discardReceiveBuffer()
{
	return 0;
}

bool
Usart2::hasError()
{
	return UsartHal2::getInterruptFlags().any(
		UsartHal2::InterruptFlag::ParityError |
#ifdef USART_ISR_NE
		UsartHal2::InterruptFlag::NoiseError |
#endif
		UsartHal2::InterruptFlag::OverrunError | UsartHal2::InterruptFlag::FramingError);
}
void
Usart2::clearError()
{
	return UsartHal2::acknowledgeInterruptFlags(
		UsartHal2::InterruptFlag::ParityError |
#ifdef USART_ISR_NE
		UsartHal2::InterruptFlag::NoiseError |
#endif
		UsartHal2::InterruptFlag::OverrunError | UsartHal2::InterruptFlag::FramingError);
}

}	// namespace modm::platform

MODM_ISR(USART2)
{
	using namespace modm::platform;
	if (UsartHal2::isTransmitRegisterEmpty()) {
		if (txBuffer.isEmpty()) {
			// transmission finished, disable TxEmpty interrupt
			UsartHal2::disableInterrupt(UsartHal2::Interrupt::TxEmpty);
		}
		else {
			UsartHal2::write(txBuffer.get());
			txBuffer.pop();
		}
	}
	UsartHal2::acknowledgeInterruptFlags(UsartHal2::InterruptFlag::OverrunError);
}
