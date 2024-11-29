#include "TensorLib/Include/Base/Tensor/Functional.h"
#include <iostream>

int main()
{
	using namespace DragonianLib;
	auto Tensor1 = Functional::Empty(IDim(1, 1, 4, 5, 1, 4, 1, 9, 1, 9));
	auto Tensor2 = Functional::Empty(IDim(1, 1, 4, 5, 1, 4, 1, 9, 1, 9));
	auto Tensor3 = Functional::EmptyLike(Tensor1);
	auto Tensor4 = Functional::Randn(IDim(1, 18)).UnSqueeze(-1);
	Tensor4.Eval();
	for(auto i : Tensor4)
		for (auto j : i)
			for (auto k : j)
				std::cout << k << " ";
	std::cout << '\n';
	//Tensor4 = Tensor4.Slice(IRanges(Range{ -1ll , -2ll, -19ll }, None, None));
	Tensor4.Eval();
	for (auto i : Tensor4)
		for (auto j : i)
			for (auto k : j)
				std::cout << k << " ";
	//Tensor4.PowInplace(Tensor4);
	Tensor4 = Tensor4.Log();
	auto Tensor5 = Tensor4 < 1.f;
	auto Tensor6 = Tensor4 < 0.5f;
	auto Tensor7 = Tensor5 && Tensor6;
	auto Tensor8 = Tensor5 || Tensor6;
	std::cout << '\n';
	for (auto i : Tensor4)
		for (auto j : i)
			for (auto k : j)
				std::cout << k << " ";
	std::cout << '\n';
	for (auto i : Tensor5)
		for (auto j : i)
			for (auto k : j)
				std::cout << k << " ";
	std::cout << '\n';
	for (auto i : Tensor6)
		for (auto j : i)
			for (auto k : j)
				std::cout << k << " ";
	std::cout << '\n';
	Tensor7.Eval();
	for (auto i : Tensor7)
		for (auto j : i)
			for (auto k : j)
				std::cout << k << " ";
	std::cout << '\n';
	Tensor8.Eval();
	for (auto i : Tensor8)
		for (auto j : i)
			for (auto k : j)
				std::cout << k << " ";
	std::cout << '\n';
	Tensor1 = Tensor4;
	Tensor1.Eval();
	Tensor1 = Tensor1.Transpose(1, 0);
	Tensor2 = Tensor2.Transpose(1, 0);
	//Tensor2 = Tensor1;
	SetRandomSeed(114);
	Tensor1.RandnFix(0, 1);
	SetRandomSeed(114);
	Tensor2.RandnFix(0, 1);
	Tensor1.MakeContinuous();
	Tensor2 = Tensor2.Transpose(1, 0);
	Tensor1.Eval();
	Tensor2.Eval();
	Tensor3.Eval();
}
