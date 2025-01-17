#include "TensorLib/Include/Base/Tensor/Functional.h"
#include "TensorLib/Include/Base/Module/Convolution.h"
#include "TensorLib/Include/Base/Module/Embedding.h"
#include "TensorLib/Include/Base/Module/Linear.h"
#include <iostream>

auto MyLastTime = std::chrono::high_resolution_clock::now();
size_t TotalStep = 0;
void ShowProgressBar(size_t progress) {
	int barWidth = 70;
	float progressRatio = static_cast<float>(progress) / float(TotalStep);
	int pos = static_cast<int>(float(barWidth) * progressRatio);

	std::cout << "\r";
	std::cout.flush();
	auto TimeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - MyLastTime).count();
	MyLastTime = std::chrono::high_resolution_clock::now();
	std::cout << "[Speed: " << 1000.0f / static_cast<float>(TimeUsed) << " it/s] ";
	std::cout << "[";
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progressRatio * 100.0) << "%  ";
}

void ProgressCb(size_t a, size_t b)
{
	if (a == 0)
		TotalStep = b;
	ShowProgressBar(a);
}

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

struct my_struct
{
	int	a = 0;
	int	b = 0;
	int c = 0;
};

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

	TypeTraits::MemberCountOf<my_struct>();

	auto Weight = Functional::Linspace(0.f, 100.f, 2048ll * 100);
	auto Embedding = Weight.View(100, -1);
	Embedding.Eval();
	int array[][1] = { {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20} };
	auto Indice = Functional::CopyFromArrayLike(array).EvalMove().View(-1, 4);
	std::cout << Functional::Stack<4, DMIODLETT(Indice)>({ Indice, Indice, Indice, Indice }, 1).Eval();

	std::cout << "\n";
	system("pause");

}
