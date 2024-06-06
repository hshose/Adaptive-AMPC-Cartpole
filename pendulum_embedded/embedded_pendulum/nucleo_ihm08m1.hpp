/*
 * Copyright (C) 2019 Raphael Lehmann <raphael@rleh.de>
 * Copyright (C) 2021 Christopher Durand
 * Copyright (C) 2023 Henrik Hose
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HARDWARE_V2_HPP
#define HARDWARE_V2_HPP

#include <modm/platform.hpp>

#include <librobots2/motor/motor_bridge.hpp>

#include <modm/board.hpp>

using namespace modm::platform;

namespace Board
{
using namespace modm::literals;

struct Motor {

	// Nucleo header pins
	using C7_28  =  GpioA0;
	using C7_34  =  GpioB0;
	using C7_36  =  GpioC1;
	using C7_38  =  GpioC0;

	using C10_15 =  GpioA7;
	using C10_21 =  GpioA9;
	using C10_23 =  GpioA8;
	using C10_24 =  GpioB1;
	using C10_33 =  GpioA10;

    using PhaseUP		= C10_23;
    using PhaseUN		= C10_15;
    using PhaseVP		= C10_21;
    using PhaseVN		= C7_34;
    using PhaseWP		= C10_33;
    using PhaseWN		= C10_24;

	using MotorTimer	= Timer1;

	using Phase = librobots2::motor::Phase;
	using PhaseConfig = librobots2::motor::PhaseConfig;
	using BridgeConfig = librobots2::motor::BridgeConfig;

	static constexpr uint16_t MaxPwm{1023}; // 11 bit PWM

	static inline void
	setCompareValue(uint16_t compareValue)
	{
		MotorTimer::setCompareValue(1, compareValue);
		MotorTimer::setCompareValue(2, compareValue);
		MotorTimer::setCompareValue(3, compareValue);
	}

	static inline void
	setCompareValue(Phase phase, uint16_t compareValue)
	{
		if (phase == Phase::PhaseU) {
			MotorTimer::setCompareValue(1, compareValue);
		} else if (phase == Phase::PhaseV) {
			MotorTimer::setCompareValue(2, compareValue);
		} else {
			MotorTimer::setCompareValue(3, compareValue);
		}
	}

	static void
	configure(Phase phase, PhaseConfig phaseOutputConfig)
	{
		switch(phaseOutputConfig) {
			case PhaseConfig::HiZ:
				MotorTimer::configureOutputChannel(static_cast<uint32_t>(phase) + 1,
						MotorTimer::OutputCompareMode::ForceActive,
						MotorTimer::PinState::Enable,
						MotorTimer::OutputComparePolarity::ActiveLow,
						MotorTimer::PinState::Enable,
						MotorTimer::OutputComparePolarity::ActiveLow,
						MotorTimer::OutputComparePreload::Disable
						);
				break;
			case PhaseConfig::Pwm:
				MotorTimer::configureOutputChannel(static_cast<uint32_t>(phase) + 1,
						MotorTimer::OutputCompareMode::Pwm,
						MotorTimer::PinState::Enable,
						MotorTimer::OutputComparePolarity::ActiveHigh,
						MotorTimer::PinState::Enable,
						MotorTimer::OutputComparePolarity::ActiveLow,
						MotorTimer::OutputComparePreload::Disable
						);
				break;
			case PhaseConfig::High:
				MotorTimer::configureOutputChannel(static_cast<uint32_t>(phase) + 1,
						MotorTimer::OutputCompareMode::ForceActive,
						MotorTimer::PinState::Enable,
						MotorTimer::OutputComparePolarity::ActiveHigh,
						MotorTimer::PinState::Enable,
						MotorTimer::OutputComparePolarity::ActiveHigh,
						MotorTimer::OutputComparePreload::Disable
						);
				break;
			case PhaseConfig::Low:
				MotorTimer::configureOutputChannel(static_cast<uint32_t>(phase) + 1,
						MotorTimer::OutputCompareMode::ForceActive,
						MotorTimer::PinState::Enable,
						MotorTimer::OutputComparePolarity::ActiveLow,
						MotorTimer::PinState::Enable,
						MotorTimer::OutputComparePolarity::ActiveHigh,
						MotorTimer::OutputComparePreload::Disable
						);
				break;
		}
	}

	static inline void
	configure(const BridgeConfig& config)
	{
		configure(Phase::PhaseU, config.config[0]);
		configure(Phase::PhaseV, config.config[1]);
		configure(Phase::PhaseW, config.config[2]);
	}

	static inline void
	configure(PhaseConfig config)
	{
		configure(Phase::PhaseU, config);
		configure(Phase::PhaseV, config);
		configure(Phase::PhaseW, config);
	}
	
	static void
	initialize()
	{
		MotorTimer::enable();
		//MotorTimer::setMode(MotorTimer::Mode::UpCounter);
		MotorTimer::setMode(MotorTimer::Mode::CenterAligned1);

		// MotorTimer clock: APB2 timer clock (170MHz)
		MotorTimer::setPrescaler(1);
		// Prescaler: 1 -> Timer counter frequency: 170MHz
		MotorTimer::setOverflow(MaxPwm);
		// Pwm frequency: 170MHz / 2048 / 2 = 83kHz

		configure(Phase::PhaseU, PhaseConfig::HiZ);
		configure(Phase::PhaseV, PhaseConfig::HiZ);
		configure(Phase::PhaseW, PhaseConfig::HiZ);

		setCompareValue(0);

		MotorTimer::applyAndReset();

		// repetition counter = 1
		// only trigger interrupt on timer underflow in center-aligned mode
		// must be set directly after starting the timer
		TIM1->RCR = 1;
		// 0b1101: "tim_oc4refc rising or tim_oc6refc falling edges generate pulses on tim_trgo2"
		// 0111: Compare - tim_oc4refc signal is used as trigger output (tim_trgo2)
		TIM1->CR2 |= (0b0111 << TIM_CR2_MMS2_Pos);

		MotorTimer::configureOutputChannel(4, MotorTimer::OutputCompareMode::Pwm, int(MaxPwm*0.95));

		// MotorTimer::enableInterruptVector(MotorTimer::Interrupt::Update, true, 5);
		// MotorTimer::enableInterrupt(MotorTimer::Interrupt::Update);

		MotorTimer::enableOutput();

		MotorTimer::pause();

		MotorTimer::connect<PhaseUN::Ch1n,
							PhaseVN::Ch2n,
							PhaseWN::Ch3n,
							PhaseUP::Ch1,
							PhaseVP::Ch2,
							PhaseWP::Ch3>();
	}
};

struct EncoderCart {

	using C7_17  = GpioA15;
	using C10_31 = GpioB3;

	using PinA			= C7_17;
	using PinB			= C10_31;

	using Timer			= Timer2;

	static inline
	Timer::Value getEncoderRaw()
	{
		return Timer::getValue();
	}

	static void
	initialize()
	{
		// PinA::setInput(Gpio::InputType::Floating);
		// PinB::setInput(Gpio::InputType::Floating);
		Timer::enable();
		Timer::setMode(Timer::Mode::UpCounter, Timer::SlaveMode::Encoder3);
		// Overflow must be 16bit because else a lot of our motor control code will break!
		Timer::setOverflow(0xffff);

		Timer::connect<PinA::Ch1, PinB::Ch2>();

		Timer::start();
	}
};

struct EncoderPendulum {
	using C10_19 = GpioC7;
	using C10_4  = GpioC6;
	
	using PinA			= C10_4;
	using PinB			= C10_19;

	using Timer			= Timer3;

	static inline
	Timer::Value getEncoderRaw()
	{
		return Timer::getValue();
	}

	static inline
	void initialize()
	{
		PinA::setInput(Gpio::InputType::Floating);
		PinB::setInput(Gpio::InputType::Floating);
		Timer::enable();
		Timer::setMode(Timer::Mode::UpCounter, Timer::SlaveMode::Encoder3);
		// Overflow must be 16bit because else a lot of our motor control code will break!
		Timer::setOverflow(0xffff);

		Timer::connect<PinA::Ch1, PinB::Ch2>();

		Timer::start();
	}
};

namespace EncoderTimer{
	using Timer = Timer20;
	constexpr uint16_t 	input_capture_overflow 		= 42500;
	constexpr float 	input_capture_freq_hz 		= 1000.00;	// Hz
	constexpr uint16_t 	input_capture_prescaler 	= SystemClock::Frequency / input_capture_freq_hz / input_capture_overflow;
	// constexpr float 	input_capture_ms_per_tick 	= ( 1.0f / input_capture_freq_hz ) * 1000.0;
	static inline void
	initialize()
	{
		// Timer::connect<>();
		Timer::enable();
		Timer::setMode(Timer::Mode::UpCounter);
		Timer::setPrescaler(input_capture_prescaler);
		Timer::setOverflow(input_capture_overflow);
		// Timer::configureInputChannel<GpioC2::Ch2>(Timer::InputCaptureMapping::InputOwn,
										// interrupt_prescaler, Timer::InputCapturePolarity::Rising, 0, false);
		Timer::enableInterruptVector(Timer::Interrupt::CaptureCompare2, true, 10);
		Timer::enableInterrupt(Timer::Interrupt::CaptureCompare2);
		Timer::applyAndReset();
		Timer::start();
	}
}

inline void
initializeAllPeripherals()
{
	Motor::initialize();
	EncoderPendulum::initialize();
	EncoderCart::initialize();
	// EncoderTimer::initialize();
}

}

#endif
