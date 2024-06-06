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

#ifndef ENCODER_HPP
#define ENCODER_HPP


#include <numbers>
#include <atomic>

template<class EncoderBridge>
class Encoder{
	std::atomic<int32_t> angle_raw = 0;
	uint16_t last_val = 0;

	float x{0}, v{0}, a{0};
	std::atomic<float> v_out{0};
	const float radperticks{2.f*std::numbers::pi_v<float>/4096.f};
	const float scaling, offset;
	const float dt_;
	const float Kx, Kv, Ka;

public:

	Encoder(
		const float scaling=1, 
		const float offset = 0,
		const float dt = 50e-3,
		const float Kx = 1,
		const float Kv = 30, 
		const float Ka = 500
		)
		: scaling(scaling), offset(offset), dt_(dt), Kx(Kx), Kv(Kv), Ka(Ka)
		{};

	inline void
	update(){
		update(dt_);
	}

	inline
	void update(const float dt){
		// Overflow is 0xffff
		const uint16_t val = static_cast<int32_t>(EncoderBridge::getEncoderRaw());
		const int16_t diff = static_cast<int16_t>(val - last_val);
		angle_raw += diff;
		last_val = val;

		// predict
		x += dt*v + dt*dt/2.f*a;
		v += dt*a;

		// update
		const auto dz = radperticks*float{angle_raw}-x;
		x += Kx*dz;
		v += Kv*dz;
		a += Ka*dz;

		v_out.store(v, std::memory_order_relaxed);
	}

	inline float get_x(){ return scaling*radperticks*float{angle_raw.load(std::memory_order_relaxed)}+offset; };
	inline float get_v(){ return scaling*v_out.load(std::memory_order_relaxed); };
	inline float get_a(){ return scaling*a; };
};

#endif // ENCODER_HPP