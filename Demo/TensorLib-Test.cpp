#include "TensorLib/Include/Base/Tensor/Functional.h"
#include <iostream>

int main()
{
	using namespace DragonianLib;
	auto Tensor1 = Functional::Randn<Complex32>(IDim(1, 9, 1, 9)).Eval();
	auto Tensor2 = Functional::Randn<bool>(IDim(1, 9, 1, 9)).Eval();
	for (auto i : Tensor1)
		for (auto j : i)
			for (auto k : j)
				for (auto l : k)
					std::cout << l << " ";
	std::cout << "\n\n";
	for (size_t i = 0; i < 1; i++)
		for (size_t j = 0; j < 9; j++)
			for (size_t k = 0; k < 1; k++)
				for (size_t l = 0; l < 9; l++)
					std::cout << Tensor1(i, j, k, l) << " ";
	std::cout << "\n\n";
	for (auto i : Tensor1)
		for (auto j : i)
			for (auto k : j)
				for (auto l : k)
					std::cout << std::tan(l) << " ";
	std::cout << "\n\n";
	Tensor1 = Tensor1.Tan().EvalMove();
	for (size_t i = 0; i < 1; i++)
		for (size_t j = 0; j < 9; j++)
			for (size_t k = 0; k < 1; k++)
				for (size_t l = 0; l < 9; l++)
					std::cout << Tensor1(i, j, k, l) << " ";
	std::cout << "\n\n";
}
