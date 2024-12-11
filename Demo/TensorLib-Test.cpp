#include "TensorLib/Include/Base/Tensor/Functional.h"
#include "Libraries/NumpySupport/NumpyFileFormat.h"
#include "OnnxLibrary/TextToSpeech/Modules/Models/Header/FishSpeech.hpp"
#include <iostream>
#include "TensorLib/Include/Base/Module/Convolution.h"
#include "TensorLib/Include/Base/Module/Embedding.h"
#include "TensorLib/Include/Base/Module/Linear.h"

class MyModule : public DragonianLib::Graph::Module
{
public:
	MyModule() : Module(nullptr, L"MyModule"),
		DragonianLibRegisterLayer(_List),
		DragonianLibRegisterLayer(_Seq)
	{
		using emb = DragonianLib::Graph::Embedding<float, DragonianLib::Device::CPU>;
		using linear = DragonianLib::Graph::Linear<float, DragonianLib::Device::CPU>;
		using conv1d = DragonianLib::Graph::Conv1D<float, DragonianLib::Device::CPU>;
		_List.Append(
			DragonianLibLayerItem(
				emb,
				DragonianLib::Graph::EmbeddingParam{ 1919, 810 }
			)
		);
		_List.Append(
			DragonianLibLayerItem(
				linear,
				DragonianLib::Graph::LinearParam{ 514, 114 }
			)
		);
		_List.Append(
			DragonianLibLayerItem(
				conv1d,
				DragonianLib::Graph::Conv1DParam{ 114, 514, 9 }
			)
		);
	}
private:
	DragonianLib::Graph::ModuleList _List;
	DragonianLib::Graph::Sequential _Seq;
};

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

template <typename Fn>
void WithTimer(const Fn& fn)
{
	auto start = std::chrono::high_resolution_clock::now();
	fn();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
}

int main()
{
	using namespace DragonianLib;

	SetWorkerCount(16);
	SetMaxTaskCountPerOperator(16);

	auto Weight = Functional::Linspace(0.f, 100.f, 2048ll * 100);
	auto Embedding = Weight.View(100, -1);
	Embedding.Eval();
	int array[][1] = { {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20} };
	auto Indice = Functional::CopyFromArrayLike(array).EvalMove().View(-1, 4);
	std::cout << Functional::Stack<4, DMIODLETT(Indice)>({ Indice, Indice, Indice, Indice }, 1).Eval();
}
