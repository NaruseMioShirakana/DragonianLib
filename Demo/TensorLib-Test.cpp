#include "TensorLib/Include/Base/Tensor/Functional.h"
#include "Libraries/NumpySupport/NumpyFileFormat.h"
#include "OnnxLibrary/TextToSpeech/Modules/Models/Header/FishSpeech.hpp"
#include <iostream>

struct Integer
{
	int i = 0;
	operator std::string() const { return std::to_string(i); }
};

template <typename T>
std::enable_if_t<std::is_same_v<T, Integer>, std::string> DragonianLibCvtToString(const T& t)
{
	return std::to_string(t.i);
}

int main()
{
	using namespace DragonianLib;
	auto Tensor1 = Functional::Randn<float>(IDim(1, 9, 1, 9)).EvalMove();
	Functional::NumpySave(LR"(D:\114514.npy)", Tensor1);
	auto Tensor2 = Functional::NumpyLoad<float, 4>(LR"(D:\114514.npy)");
	auto Tensor3 = Functional::Linspace(0.f, 0.36f, 6);
	auto Tensor4 = Tensor3.View(3, -1);
	std::cout << Tensor4.Padding({ {{2, 1}} }, PaddingType::Reflect, 1.f).Eval().CastToString(false) << "\n\n";
	std::cout << Tensor4 << "\n\n";
	std::cout << Tensor4.Shape() << "\n\n";
	Tensor2 = Functional::Permute(Tensor2, 0, 3, 2, 1);
	std::cout << Tensor2 << "\n\n";
	std::cout << Tensor1 << "\n\n";
	std::cout << (Tensor2 == Tensor1).Eval() << "\n\n";
	std::cout << Tensor1 << "\n\n";
	(Tensor1 += Tensor1).Eval();
	std::cout << Tensor1 << "\n\n";
	Tensor1 = Tensor1.Tan().EvalMove();
	std::cout << Tensor1 << "\n\n";
}
