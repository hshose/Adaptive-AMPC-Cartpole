/*
 * Copyright (c) 2021, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#pragma once
#include <cmath>

/// @cond
namespace modm::platform
{

void delay_ns(uint32_t ns);

constexpr uint16_t
computeDelayNsPerLoop(uint32_t hz)
{
	return std::round(3'000'000'000.0 / hz);
}

}
/// @endcond

