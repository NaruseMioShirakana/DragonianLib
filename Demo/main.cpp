#include "Tensor/Tensor.h"
#include <iostream>
#include <windows.h>
#include "Tensor/Float32Tensor.h"

template<typename _T = float>
void PrintTensor(libsvc::Tensor& _Tensor)
{
	for (libsvc::SizeType i = 0; i < _Tensor.Size(0); ++i)
	{
		std::cout << '[';
		for (libsvc::SizeType j = 0; j < _Tensor.Size(1); ++j)
			std::cout << _Tensor.Item<_T>({ i,j }) << ", ";
		std::cout << "],\n";
	}
	std::cout << "\n";
}

template <>
void PrintTensor<bool>(libsvc::Tensor& _Tensor)
{
	for (libsvc::SizeType i = 0; i < _Tensor.Size(0); ++i)
	{
		std::cout << '[';
		for (libsvc::SizeType j = 0; j < _Tensor.Size(1); ++j)
			std::cout << ((_Tensor.Item<bool>({ i,j })) ? "true " : "false") << ", ";
		std::cout << "]\n";
	}
	std::cout << "\n";
}

int main()
{
	auto VectorA = libsvc::VectorArangeImpl(0, 20000, 233.f);
	libsvc::Vector<float> VectorB(40000);
	int64_t AShape[] = { 20000 }, BShape[] = { 40000 }, Stride[] = { 1 }, Step[] = { 1 }, Begin[] = { 0 };

	auto time = clock();
	for (int i = 0; i < 768; ++i)
	{
		libsvc::Linear1DImpl(
		   VectorB.data(), BShape, Stride, Begin, Step, 
		   VectorA.data(), AShape, Stride, Begin, Step
	   );
	}
	std::cout << clock() - time;

	libsvc::Tensor aaaaaaaaaaaaa{ {114,514,810}, libsvc::TensorType::Float32 };
	aaaaaaaaaaaaa.Assign(1.f);
	aaaaaaaaaaaaa.Permute({2,0,1}).Assign(1.f);

	libsvc::Tensor::SetThreadCount(8);
	libsvc::Tensor::EnableTimeLogger(true);
	libsvc::ThreadPool Thp;
	Thp.EnableTimeLogger(true);
	Thp.Init(8);
	libsvc::Tensor::Arange(1., 5., 0.3).UnSqueeze(0).Invoke(1, PrintTensor);
	constexpr float Temp[10]{ 114,514,1919,810,1453,721,996,7,1919,810 };
	libsvc::Tensor Ten({ 3,5 }, libsvc::TensorType::Float32);
	Ten.RandnFix();
	std::cout << "\nTen:\n";
	Ten.Invoke(1, PrintTensor);
	std::cout << "\nGather Op Test\n";
	auto Indices = libsvc::Tensor::ConstantOf({ 2,2 }, 0ll, libsvc::TensorType::Int64);
	Indices[0][0] = 0ll; Indices[0][1] = 1ll; Indices[1][0] = 2ll; Indices[1][1] = 1ll;
	Indices.Invoke(1, PrintTensor<libsvc::SizeType>);
	Ten.Gather(Indices, 1).Invoke(1, PrintTensor);
	std::cout << "\nDiff Op Test\n";
	libsvc::Tensor::Diff(Ten, 0, nullptr).UnSqueeze(0).Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten = libsvc::Tensor::Stack({ Ten ,Ten, Ten }, 0);
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten += Ten;
	Ten.Sin_();
	std::cout << "Op Test: \n";
	Ten.Invoke(1, PrintTensor);
	std::cout << "Sin Op Test: \n";
	Ten.Sin().Invoke(1, PrintTensor);
	std::cout << "Cos Op Test: \n";
	Ten.Cos().Invoke(1, PrintTensor);
	std::cout << "Tan Op Test: \n";
	Ten.Tan().Invoke(1, PrintTensor);
	std::cout << "Abs Op Test: \n";
	Ten.Abs().Invoke(1, PrintTensor);
	std::cout << "Ceil Op Test: \n";
	Ten.Ceil().Invoke(1, PrintTensor);
	std::cout << "Floor Op Test: \n";
	Ten.Floor().Invoke(1, PrintTensor);
	std::cout << "Compare Op Test: \n";
	(Ten.Abs() == Ten).Invoke(1, PrintTensor<bool>);
	(Ten.Abs() != Ten).Invoke(1, PrintTensor<bool>);
	((Ten.Abs() != Ten) + (Ten.Abs() == Ten)).Invoke(1, PrintTensor<bool>);
	std::cout << "Op Test End.\n\n\n";
	auto Tens = libsvc::Tensor::Cat({ Ten ,Ten }, 2);
	Tens.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tens = libsvc::Tensor::Cat({ Ten ,Ten }, -1);
	Tens.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tens.Cast(libsvc::TensorType::Int64).Invoke(1, PrintTensor<int64_t>);
	std::cout << '\n';
	libsvc::Tensor::Pad(Ten, { {1, 2}, {1, 2} }, libsvc::PaddingType::Reflect).Invoke(1, PrintTensor);
	std::cout << '\n';
	const libsvc::Tensor Ten114514({ 1,514,1,1919 }, libsvc::TensorType::Float32);
	LARGE_INTEGER Time1, Time2, Freq;
	QueryPerformanceFrequency(&Freq);
	Indices = libsvc::Tensor::ConstantOf({ 1000 }, 0ll, libsvc::TensorType::Int64);
	Indices[1].Assign(1ll);
	libsvc::Tensor Ten1919810({ 1,768,100000 }, libsvc::TensorType::Float32);
	const libsvc::Tensor Embedding({ 1000,768 }, libsvc::TensorType::Float32);
	Ten1919810.Fix(1.1);
	for (int64_t i = 0; i < 20; ++i)
	{
		
		Embedding[0].Assign(i);
		Embedding[1].Assign(i + 1);
		auto Emb = Embedding[0];
		QueryPerformanceCounter(&Time1);
		//auto Out = Embedding.Gather(Indices, 0, Thp);
		//Embedding.Assign(Embedding);
		//Ten114514.Permute({ 3,1,2,0 }).Clone();
		//libsvc::Tensor::Pad(Ten114514, {libsvc::None,19 },libsvc::PaddingType::Zero, libsvc::TensorType::Float32,nullptr, &Thp);
		/*libsvc::Tensor::Pad(
			Ten1919810,
			{libsvc::None,1 },
			libsvc::PaddingType::Replicate,
			libsvc::TensorType::Float32,
			nullptr, &Thp
		);*/
		//auto a = libsvc::Tensor::Diff(Ten1919810, 1, &Thp);
		//libsvc::Tensor::Stack({Ten1919810.Squeeze()}, 0, &Thp);
		//auto a = Ten1919810.Permute({ 0,2,1 });
		//libsvc::Tensor::Repeat(Ten1919810, { {0, 2} }, &Thp);
		//a.Continuous(&Thp);
		Ten1919810 = Ten1919810 * Ten1919810;
		//Thp.Commit([&]() { a.Slice({ libsvc::None,{0,192} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ libsvc::None,{192,384} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ libsvc::None,{384,572} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ libsvc::None,{572,768} }).Continuous(); });
		//Thp.Join();
		QueryPerformanceCounter(&Time2);
		std::cout << i << " CostTime:" << double(Time2.QuadPart - Time1.QuadPart) * 1000. / (double)Freq.QuadPart << "ms\n";
		//Out.Invoke(1, PrintTensor);
	}
	std::cout << "\n\n\n";
	Ten.FixOnes();
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten.Fix(114514.);
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten.Assign(Temp, sizeof(Temp));
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	auto TenA = Ten[0];
	TenA.Slice({ {0,-1 },{0,1} }).Assign(1.f);
	TenA.Slice({ {0,-1 },{1,2} }).Assign(2.f);
	TenA.Slice({ {0,-1 },{2,3} }).Assign(3.f);
	TenA.Slice({ {0,-1 },{3,4} }).Assign(4.f);
	TenA.Slice({ {0,-1 },{4,5} }).Assign(5.f);
	TenA = Ten[1];
	TenA.Slice({ {0,-1 },{0,1} }).Assign(6.f);
	TenA.Slice({ {0,-1 },{1,2} }).Assign(7.f);
	TenA.Slice({ {0,-1 },{2,3} }).Assign(8.f);
	TenA.Slice({ {0,-1 },{3,4} }).Assign(9.f);
	TenA.Slice({ {0,-1 },{4,5} }).Assign(10.f);
	TenA = TenA[2];
	TenA = TenA[4];
	TenA = TenA[0];
	TenA = TenA[0];
	auto TenB = std::move(Ten);
	TenA = std::move(TenB);
	auto TenC = TenA.Slice({ {-1, -1,-3},{0,-1 },{2,2,-1} });
	TenA.Invoke(1, PrintTensor);
	std::cout << '\n';
	TenC.Invoke(1, PrintTensor);
	std::cout << '\n';
	TenC.Assign(Temp, sizeof(Temp));
	TenC.Invoke(1, PrintTensor);
	std::cout << '\n';
	auto Tennnnn = TenA.Clone();
	TenA.Permute({ 2,0,1 }).Invoke(1, PrintTensor);
	std::cout << '\n';
	Tennnnn.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tennnnn.Clone().Invoke(1, PrintTensor);
	system("pause");
	return 0;
}