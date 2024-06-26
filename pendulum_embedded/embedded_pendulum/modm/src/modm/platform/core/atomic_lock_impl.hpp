/*
 * Copyright (c) 2009-2012, Fabian Greif
 * Copyright (c) 2010, Martin Rosekeit
 * Copyright (c) 2012, 2015, Sascha Schade
 * Copyright (c) 2012, 2015-2016, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#pragma once

#include "../device.hpp"
#include <modm/architecture/utils.hpp>
/// @cond
namespace modm::atomic
{

class Lock
{
public:
	modm_always_inline
	Lock() : cpsr(__get_PRIMASK())
	{
		__disable_irq();
	}

	modm_always_inline
	~Lock()
	{
		__set_PRIMASK(cpsr);
	}

private:
	uint32_t cpsr;
};

class Unlock
{
public:
	modm_always_inline
	Unlock() : cpsr(__get_PRIMASK())
	{
		__enable_irq();
	}

	modm_always_inline
	~Unlock()
	{
		__set_PRIMASK(cpsr);
	}

private:
	uint32_t cpsr;
};

class LockPriority
{
public:
	modm_always_inline
	LockPriority(uint32_t priority) : basepri(__get_BASEPRI())
	{
		__set_BASEPRI_MAX(priority << (8u - __NVIC_PRIO_BITS));
	}

	modm_always_inline
	~LockPriority()
	{
		__set_BASEPRI(basepri);
	}

private:
	uint32_t basepri;
};
}	// namespace modm::atomic
/// @endcond
