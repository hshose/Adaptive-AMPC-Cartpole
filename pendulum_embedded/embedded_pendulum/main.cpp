// MIT License

// Copyright (c) 2024 Henrik Hose

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <modm/platform.hpp>
#include <modm/debug/logger.hpp>
#include <modm/board.hpp>

#include <librobots2/motor/dc_motor.hpp>
#include "nucleo_ihm08m1.hpp"
#include "encoder.hpp"
#include <modm/processing/timer.hpp>

#include <neural_network_training/embedded_nn_inference/neural_network.hpp>

#include <algorithm>
#include <atomic>

namespace eni = embedded_nn_inference;

using DcMotor = librobots2::motor::DcMotor<Board::Motor>;
using EncoderPendulum = Encoder<Board::EncoderPendulum>;
using EncoderCart = Encoder<Board::EncoderCart>;
using Uart = Usart2;

struct ParamSet{
	float delta_m_add;  // additional tip mass on pole
	float delta_M;      // additional mass to cart
	float delta_C1;     // cart friction F = C1*v + C2*u 
	float delta_C2;     // motor constant F = C1*v + C2*u 
	float delta_C3;     // friction in rotary joint

	std::array<float, 5> getDeltas() const {
        return {delta_m_add, delta_C1, delta_M, delta_C2, delta_C3};
    }
};

constexpr ParamSet mpi_tuned{0.01, 0.5, -1.422285704372642, 0.5073308828538601, 0.010};
constexpr ParamSet quanser_tuned{-0.020395683847472465, 0.368, 0.17509906126238262, -0.8208464290203645, 0.002665466159162661};

constexpr ParamSet all_zero{0,0,0,0,0};

constexpr std::array parameter{quanser_tuned};

EncoderPendulum encoderPendulum(-1, std::numbers::pi_v<float>);
EncoderCart encoderCart(-0.01483);
std::atomic<float> motorVoltage{0};

int
main()
{
    Board::initialize();
	Board::initializeAllPeripherals();

	Uart::connect<GpioOutputA2::Tx>();
	Uart::connect<GpioInputA3::Rx>();
	Uart::initialize<Board::SystemClock, 921600_Bd>();

	Board::Motor::setCompareValue(0);
	Board::Motor::MotorTimer::start();
	Board::Motor::configure(Board::Motor::PhaseConfig::Low);

	DcMotor motor;

	modm::PeriodicTimer control_loop_timer(1ms);
	modm::Timeout starting_timer;
	modm::Timeout running_timer;

	const auto setpoint_timeout{50ms};
	const float setpoint_timeout_seconds{std::chrono::duration_cast<std::chrono::duration<float>>(setpoint_timeout).count()};
	modm::PeriodicTimer setpoint_timer(setpoint_timeout);

	auto parameter_it = parameter.begin();
	bool last_button_state = false;
	size_t loop = 0;

	bool running = false;

	while (1)
	{	

		Board::LedD13::toggle();
		if(!Board::Button::read()){
			starting_timer.restart(5s);
		}
	


		if (setpoint_timer.execute()){
			encoderCart.update(50e-3);
			encoderPendulum.update(50e-3);


			auto start = modm::PreciseClock::now();
			auto pend_ang = encoderPendulum.get_x();

			pend_ang = std::fmod(pend_ang+std::numbers::pi_v<float>, 2.f*std::numbers::pi_v<float>)-std::numbers::pi_v<float>;

			const std::array<float, 4> x{encoderCart.get_x(), pend_ang, encoderCart.get_v(),encoderPendulum.get_v()};

			const auto x_norm 	= eni::u::normalize_input(x);
			const auto u_norm 	= eni::u::call_nn(x_norm);
			const auto u_den 	= eni::u::denormalize_output(u_norm);
			const auto J_norm 	= eni::J::call_nn(x_norm);
			const auto J_den 	= eni::J::denormalize_output(J_norm);

			auto linear_predictor = [](float u_nom, const std::array<float, 5>& J, const std::array<float, 5> delta_parameter) {
				for (size_t i = 0; i < J.size(); i++){
					u_nom += J.at(i)*delta_parameter.at(i);
				}
				return std::clamp(u_nom, -9.f, 9.f);
			};


			const auto u = linear_predictor(u_den.at(0), J_den, parameter_it->getDeltas());
			const auto duration = modm::PreciseClock::now() - start;
			
			if (starting_timer.execute()){
				running_timer.restart(30s);
				running = true;
			}

			if (running_timer.execute()){
				running = false;
			}

			if(running){
				motorVoltage.store(u, std::memory_order_relaxed);
			}
			else{
				motorVoltage.store(0, std::memory_order_relaxed);	
			}
			constexpr float maxMotorVolt = 10;
			int16_t setpoint = static_cast<int16_t>(std::numeric_limits<int16_t>::max() * (motorVoltage/maxMotorVolt));
			motor.setSetpoint(setpoint);
		}

		if(control_loop_timer.execute()){
			MODM_LOG_INFO
				<< encoderCart.get_x() << ","
				<< encoderPendulum.get_x() << ","
				<< encoderCart.get_v() << ","
				<< encoderPendulum.get_v() << ","
				<< motorVoltage.load(std::memory_order_relaxed) << modm::endl;
		}
	}

	return 0;
}