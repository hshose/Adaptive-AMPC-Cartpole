/*
 * Copyright (c) 2013-2014, Kevin Läufer
 * Copyright (c) 2014, Sascha Schade
 * Copyright (c) 2014, 2016-2017, Niklas Hauser
 * Copyright (c) 2023, Christopher Durand
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_STM32F3_ADC2_HPP
#	error 	"Don't include this file directly, use 'adc_2.hpp' instead!"
#endif

#include <modm/architecture/interface/delay.hpp>	// modm::delayMicroseconds
#include <modm/platform/clock/rcc.hpp>

void
modm::platform::Adc2::initialize(const ClockMode clk,
		const ClockSource clk_src,
		const Prescaler pre,
		const CalibrationMode cal, const bool blocking)
{
	uint32_t tmp = 0;

	// enable clock
	RCC->AHB2ENR |= RCC_AHB2ENR_ADC12EN;
	// select clock source
	RCC->CCIPR |= static_cast<uint32_t>(clk_src);
	// Disable deep power down
	ADC2->CR &= ~ADC_CR_DEEPPWD;
	// reset ADC
	// FIXME: not a good idea since you can only reset both
	// ADC1/ADC2 or ADC3/ADC4 at once ....

	// set ADC "analog" clock source
	if (clk != ClockMode::DoNotChange) {
		if (clk == ClockMode::Asynchronous) {
			setPrescaler(pre);
		}
		tmp  =  ADC12_COMMON->CCR;
		tmp &= ~ADC_CCR_CKMODE;
		tmp |=  static_cast<uint32_t>(clk);
		ADC12_COMMON->CCR = tmp;
	}

	// enable regulator
	ADC2->CR &= ~ADC_CR_ADVREGEN;
	ADC2->CR |= static_cast<uint32_t>(VoltageRegulatorState::Enabled);
	modm::delay_us(10);	// FIXME: this is ugly -> find better solution

	acknowledgeInterruptFlags(InterruptFlag::Ready);

	calibrate(cal, true);	// blocking calibration

	ADC2->CR |= ADC_CR_ADEN;
	if (blocking) {
		// ADEN can only be set 4 ADC clock cycles after ADC_CR_ADCAL gets
		// cleared. Setting it in a loop ensures the flag gets set for ADC
		// clocks slower than the CPU clock.
		while (not isReady()) {
			ADC2->CR |= ADC_CR_ADEN;
		}
		acknowledgeInterruptFlags(InterruptFlag::Ready);
	}
}

void
modm::platform::Adc2::disable(const bool blocking)
{
	ADC2->CR |= ADC_CR_ADDIS;
	if (blocking) {
		// wait for ADC_CR_ADDIS to be cleared by hw
		while(ADC2->CR & ADC_CR_ADDIS);
	}
	// disable clock
	RCC->AHB2ENR &= ~RCC_AHB2ENR_ADC12EN;
}

void
modm::platform::Adc2::setPrescaler(const Prescaler pre)
{
	uint32_t tmp;
	tmp  = ADC12_COMMON->CCR;
	tmp &= ~static_cast<uint32_t>(Prescaler::Div256AllBits);
	tmp |=  static_cast<uint32_t>(pre);
	ADC12_COMMON->CCR = tmp;
}

bool
modm::platform::Adc2::isReady()
{
	return static_cast<bool>(getInterruptFlags() & InterruptFlag::Ready);
}

void
modm::platform::Adc2::calibrate(const CalibrationMode mode,
									const bool blocking)
{
	if (mode != CalibrationMode::DoNotCalibrate) {
		ADC2->CR |= ADC_CR_ADCAL |
										static_cast<uint32_t>(mode);
		if(blocking) {
			// wait for ADC_CR_ADCAL to be cleared by hw
			while(ADC2->CR & ADC_CR_ADCAL);
		}
	}
}

void
modm::platform::Adc2::setLeftAdjustResult(const bool enable)
{
	if (enable) {
		ADC2->CFGR |= ADC_CFGR_ALIGN;
	} else {
		ADC2->CFGR &= ~ADC_CFGR_ALIGN;
	}
}
bool
modm::platform::Adc2::configureChannel(Channel channel,
											  SampleTime sampleTime)
{
	if (static_cast<uint8_t>(channel) > 18) {
		return false;
	}
	uint32_t tmpreg = 0;
	if (static_cast<uint8_t>(channel) < 10) {
		tmpreg = ADC2->SMPR1
			& ((~ADC_SMPR1_SMP0) << (static_cast<uint8_t>(channel) * 3));
		tmpreg |= static_cast<uint32_t>(sampleTime) <<
						(static_cast<uint8_t>(channel) * 3);
		ADC2->SMPR1 = tmpreg;
	}
	else {
		tmpreg = ADC2->SMPR2
			& ((~ADC_SMPR2_SMP10) << ((static_cast<uint8_t>(channel)-10) * 3));
		tmpreg |= static_cast<uint32_t>(sampleTime) <<
						((static_cast<uint8_t>(channel)-10) * 3);
		ADC2->SMPR2 = tmpreg;
	}
	return true;

}


bool
modm::platform::Adc2::setChannel(Channel channel,
										SampleTime sampleTime)
{
	if (!configureChannel(channel, sampleTime)) {
		return false;
	}

	ADC2->SQR1 = (static_cast<uint8_t>(channel) << ADC_SQR1_SQ1_Pos) & ADC_SQR1_SQ1_Msk;
	return true;
}

void
modm::platform::Adc2::setFreeRunningMode(const bool enable)
{
	if (enable) {
		ADC2->CFGR |=  ADC_CFGR_CONT; // set to continuous mode
	} else {
		ADC2->CFGR &= ~ADC_CFGR_CONT; // set to single mode
	}
}

void
modm::platform::Adc2::startConversion()
{
	// TODO: maybe add more interrupt flags
	acknowledgeInterruptFlags(InterruptFlag::EndOfRegularConversion |
			InterruptFlag::EndOfSampling | InterruptFlag::Overrun);
	// starts single conversion for the regular group
	ADC2->CR |= ADC_CR_ADSTART;
}

bool
modm::platform::Adc2::isConversionFinished()
{
	return static_cast<bool>(getInterruptFlags() & InterruptFlag::EndOfRegularConversion);
}

void
modm::platform::Adc2::startInjectedConversionSequence()
{
	acknowledgeInterruptFlags(InterruptFlag::EndOfInjectedConversion |
			InterruptFlag::EndOfInjectedSequenceOfConversions);

	ADC2->CR |= ADC_CR_JADSTART;
}

bool
modm::platform::Adc2::setInjectedConversionChannel(uint8_t index, Channel channel,
														  SampleTime sampleTime)
{
	if (index >= 4) {
		return false;
	}
	if (!configureChannel(channel, sampleTime)) {
		return false;
	}

	static_assert(ADC_JSQR_JSQ2_Pos == ADC_JSQR_JSQ1_Pos + 6);
	static_assert(ADC_JSQR_JSQ3_Pos == ADC_JSQR_JSQ2_Pos + 6);
	static_assert(ADC_JSQR_JSQ4_Pos == ADC_JSQR_JSQ3_Pos + 6);

	const uint32_t pos = ADC_JSQR_JSQ1_Pos + 6 * index;
	const uint32_t mask = ADC_JSQR_JSQ1_Msk << (6 * index);
	ADC2->JSQR = (ADC2->JSQR & ~mask) | (static_cast<uint32_t>(channel) << pos);
	return true;
}

template<class Gpio>
bool
modm::platform::Adc2::setInjectedConversionChannel(uint8_t index,
														  SampleTime sampleTime)
{
	return setInjectedConversionChannel(index, getPinChannel<Gpio>(), sampleTime);
}

bool
modm::platform::Adc2::setInjectedConversionSequenceLength(uint8_t length)
{
	if (length < 1 or length > 4) {
		return false;
	}
	ADC2->JSQR = (ADC2->JSQR & ~ADC_JSQR_JL)
		| ((length - 1) << ADC_JSQR_JL_Pos);
	return true;
}

bool
modm::platform::Adc2::isInjectedConversionFinished()
{
	return static_cast<bool>(getInterruptFlags() & InterruptFlag::EndOfInjectedSequenceOfConversions);
}

// ----------------------------------------------------------------------------
void
modm::platform::Adc2::enableInterruptVector(const uint32_t priority,
												const bool enable)
{
	const IRQn_Type INTERRUPT_VECTOR = ADC1_2_IRQn;
	if (enable) {
		NVIC_SetPriority(INTERRUPT_VECTOR, priority);
		NVIC_EnableIRQ(INTERRUPT_VECTOR);
	} else {
		NVIC_DisableIRQ(INTERRUPT_VECTOR);
	}
}

void
modm::platform::Adc2::enableInterrupt(const Interrupt_t interrupt)
{
	ADC2->IER |= interrupt.value;
}

void
modm::platform::Adc2::disableInterrupt(const Interrupt_t interrupt)
{
	ADC2->IER &= ~interrupt.value;
}

modm::platform::Adc2::InterruptFlag_t
modm::platform::Adc2::getInterruptFlags()
{
	return InterruptFlag_t(ADC2->ISR);
}

void
modm::platform::Adc2::acknowledgeInterruptFlags(const InterruptFlag_t flags)
{
	// Flags are cleared by writing a one to the flag position.
	// Writing a zero is ignored.
	ADC2->ISR = flags.value;
}