#include "TensorLib/Include/Base/Tensor/Functional.h"
#include <iostream>

int main()
{
	using namespace DragonianLib;
	auto Tensor1 = Functional::Randn<Complex32>(IDim(1, 9, 1, 9)).EvalMove();

	std::cout << Tensor1 << "\n\n";
	(Tensor1 += Tensor1).Eval();
	std::cout << Tensor1 << "\n\n";
	Tensor1 = Tensor1.Tan().EvalMove();
	std::cout << Tensor1 << "\n\n";
}
