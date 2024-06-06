/*
 * Copyright (c) 2013-2014, 2016, Kevin Läufer
 * Copyright (c) 2013-2017, Niklas Hauser
 * Copyright (c) 2017, Fabian Greif
 * Copyright (c) 2017, Sascha Schade
 * Copyright (c) 2018, Christopher Durand
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

#ifndef MODM_STM32_UARTHAL_2_HPP
#	error 	"Don't include this file directly, use uart_hal_2.hpp instead!"
#endif
#include <modm/platform/clock/rcc.hpp>
#include <modm/math/algorithm/prescaler.hpp>

namespace modm::platform
{

// ----------------------------------------------------------------------------
void
UsartHal2::enable()
{
	Rcc::enable<Peripheral::Usart2>();
}

void
UsartHal2::disable()
{
	// TX, RX, Uart, etc. Disable
	USART2->CR1 = 0;
	Rcc::disable<Peripheral::Usart2>();
}

void
UsartHal2::enableOperation()
{
	USART2->CR1 |= USART_CR1_UE;
}

void
UsartHal2::disableOperation()
{
	USART2->CR1 &= ~USART_CR1_UE;
}

template< class SystemClock, modm::baudrate_t baudrate, modm::percent_t tolerance >
void
UsartHal2::initialize(Parity parity, WordLength length)
{
	enable();
	disableOperation();

	constexpr uint32_t max = (1ul << 16) - 1ul;
	constexpr auto result = Prescaler::from_range(SystemClock::Usart2, baudrate, 1, max);
	modm::PeripheralDriver::assertBaudrateInTolerance< result.frequency, baudrate, tolerance >();

	uint32_t cr1 = USART2->CR1;
	// Set baudrate
	USART2->BRR = result.prescaler;
	// Set parity
	cr1 &= ~(USART_CR1_PCE | USART_CR1_PS);
	cr1 |= static_cast<uint32_t>(parity);

	// Set word length
#ifdef USART_CR1_M1
	cr1	&= ~(USART_CR1_M0 | USART_CR1_M1);
#else
	cr1	&= ~USART_CR1_M;
#endif
	cr1 |= static_cast<uint32_t>(length);

	USART2->CR1 = cr1;
}

void
UsartHal2::setSpiClock(SpiClock clk, LastBitClockPulse pulse)
{
	uint32_t cr2 = USART2->CR2;
	cr2 &= ~(USART_CR2_LBCL | USART_CR2_CLKEN);
	cr2 |= static_cast<uint32_t>(clk) | static_cast<uint32_t>(pulse);
	USART2->CR2 = cr2;
}

void
UsartHal2::setSpiDataMode(SpiDataMode mode)
{
	uint32_t cr2 = USART2->CR2;
	cr2 &= ~(USART_CR2_CPOL | USART_CR2_CPHA);
	cr2 |= static_cast<uint32_t>(mode);
	USART2->CR2 = cr2;
}
void
UsartHal2::write(uint16_t data)
{
	USART2->TDR = data;
}

void
UsartHal2::read(uint8_t &data)
{
	data = USART2->RDR;
}

void
UsartHal2::read(uint16_t &data)
{
	data = USART2->RDR;
}

void
UsartHal2::setTransmitterEnable(bool enable)
{
	if (enable) {
		USART2->CR1 |=  USART_CR1_TE;
	} else {
		USART2->CR1 &= ~USART_CR1_TE;
	}
}

void
UsartHal2::setReceiverEnable(bool enable)
{
	if (enable) {
		USART2->CR1 |=  USART_CR1_RE;
	} else {
		USART2->CR1 &= ~USART_CR1_RE;
	}
}

bool
UsartHal2::isReceiveRegisterNotEmpty()
{
	return USART2->ISR & USART_ISR_RXNE;
}

bool
UsartHal2::isTransmitRegisterEmpty()
{
	return USART2->ISR & USART_ISR_TXE;
}

bool
UsartHal2::isTransmissionComplete()
{
	return USART2->ISR & USART_ISR_TC;
}

void
UsartHal2::enableInterruptVector(bool enable, uint32_t priority)
{
	if (enable) {
		// Set priority for the interrupt vector
		NVIC_SetPriority(USART2_IRQn, priority);

		// register IRQ at the NVIC
		NVIC_EnableIRQ(USART2_IRQn);
	}
	else {
		NVIC_DisableIRQ(USART2_IRQn);
	}
}

void
UsartHal2::setInterruptPriority(uint32_t priority)
{
	NVIC_SetPriority(USART2_IRQn, priority);
}

void
UsartHal2::enableInterrupt(Interrupt_t interrupt)
{
	USART2->CR1 |= interrupt.value;
}

void
UsartHal2::disableInterrupt(Interrupt_t interrupt)
{
	USART2->CR1 &= ~interrupt.value;
}

UsartHal2::InterruptFlag_t
UsartHal2::getInterruptFlags()
{
	return InterruptFlag_t( USART2->ISR );
}

void
UsartHal2::acknowledgeInterruptFlags(InterruptFlag_t flags)
{
	// Flags are cleared by writing a one to the flag position.
	// Writing a zero is (hopefully) ignored.
	USART2->ICR = flags.value;
}

} // namespace modm::platform