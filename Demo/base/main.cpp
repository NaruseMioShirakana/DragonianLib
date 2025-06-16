#include <iostream>

#include "TensorLib/Include/Base/Tensor/Functional.h"

#ifndef DRAGONIANLIB_USE_SHARED_LIBS

#include "TensorLib/Include/Base/Tensor/Tensor.h"

#endif

int main()
{
#ifndef DRAGONIANLIB_USE_SHARED_LIBS
	const int size = 10;
	return 0;
#endif
}