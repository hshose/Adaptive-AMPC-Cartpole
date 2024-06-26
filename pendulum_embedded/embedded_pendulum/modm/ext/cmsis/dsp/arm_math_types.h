/*
 * Copyright (c) 2021, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef _ARM_MATH_TYPES_WRAPPER_H_
#define _ARM_MATH_TYPES_WRAPPER_H_

#define ARM_MATH_CM4
#define __ARM_FEATURE_MVE 0

#ifndef __FPU_PRESENT
#define __FPU_PRESENT 1
#endif
/* Local configuration file */
#if __has_include(<arm_math_local.h>)
#  include <arm_math_local.h>
#endif

#include "arm_math_types_internal.h"

#endif /* ifndef _ARM_MATH_TYPES_WRAPPER_H_ */