#include "Tensor/Tensor.h"
#include <iostream>
#include <windows.h>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#include "Tensor/Float32Tensor.h"
#include "AvCodec.h"

template<typename _T = float>
void PrintTensor(DragonianLib::Tensor& _Tensor)
{
	for (DragonianLib::SizeType i = 0; i < _Tensor.Size(0); ++i)
	{
		std::cout << '[';
		for (DragonianLib::SizeType j = 0; j < _Tensor.Size(1); ++j)
			std::cout << _Tensor.Item<_T>({ i,j }) << ", ";
		std::cout << "],\n";
	}
	std::cout << "\n";
}

template <>
void PrintTensor<bool>(DragonianLib::Tensor& _Tensor)
{
	for (DragonianLib::SizeType i = 0; i < _Tensor.Size(0); ++i)
	{
		std::cout << '[';
		for (DragonianLib::SizeType j = 0; j < _Tensor.Size(1); ++j)
			std::cout << ((_Tensor.Item<bool>({ i,j })) ? "true " : "false") << ", ";
		std::cout << "]\n";
	}
	std::cout << "\n";
}

void Demo()
{
	DragonianLib::Tensor aaaaaaaaaaaaa{ {114,514,810}, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU };
	aaaaaaaaaaaaa.Assign(1.f);
	aaaaaaaaaaaaa.Permute({ 2,0,1 }).Assign(1.f);

	DragonianLib::Tensor::SetThreadCount(8);
	DragonianLib::Tensor::EnableTimeLogger(false);
	DragonianLib::ThreadPool Thp;
	Thp.EnableTimeLogger(false);
	Thp.Init(8);
	DragonianLib::Tensor::Arange(1., 5., 0.3).UnSqueeze(0).Invoke(1, PrintTensor);
	constexpr float Temp[10]{ 114,514,1919,810,1453,721,996,7,1919,810 };
	DragonianLib::Tensor Ten({ 3,5 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);
	Ten.RandnFix();
	auto a = Ten;
	std::cout << "\nTen:\n";
	Ten.Invoke(1, PrintTensor);
	std::cout << "\nGather Op Test\n";
	auto Indices = DragonianLib::Tensor::ConstantOf({ 2,2 }, 0ll, DragonianLib::TensorType::Int64);
	Indices[0][0] = 0ll; Indices[0][1] = 1ll; Indices[1][0] = 2ll; Indices[1][1] = 1ll;
	Indices.Invoke(1, PrintTensor<DragonianLib::SizeType>);
	Ten.Gather(Indices, 1).Invoke(1, PrintTensor);
	std::cout << "\nCumSum Op Test\n";
	DragonianLib::Tensor::CumSum(Ten, 0, nullptr).UnSqueeze(0).Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten = DragonianLib::Tensor::Stack({ Ten ,Ten, Ten }, 0);
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
	auto Tens = DragonianLib::Tensor::Cat({ Ten ,Ten }, 2);
	Tens.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tens = DragonianLib::Tensor::Cat({ Ten ,Ten }, -1);
	Tens.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tens.Cast(DragonianLib::TensorType::Int64).Invoke(1, PrintTensor<int64_t>);
	std::cout << '\n';
	DragonianLib::Tensor::Pad(Ten, { {1, 2}, {1, 2} }, DragonianLib::PaddingType::Reflect).Invoke(1, PrintTensor);
	std::cout << '\n';
	const DragonianLib::Tensor Ten114514({ 1,514,1,1919 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);
	LARGE_INTEGER Time1, Time2, Freq;
	QueryPerformanceFrequency(&Freq);
	Indices = DragonianLib::Tensor::ConstantOf({ 1000 }, 0ll, DragonianLib::TensorType::Int64);
	Indices[1].Assign(1ll);
	const DragonianLib::Tensor Embedding({ 1000,768 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);

	for (int64_t i = 0; i < 20; ++i)
	{
		DragonianLib::Tensor Ten1919810({ 1,768,100000 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);
		Ten1919810.RandFix(&Thp);
		Embedding[0].Assign(i);
		Embedding[1].Assign(i + 1);
		//Ten1919810.Assign(i);
		auto Emb = Embedding[0];
		QueryPerformanceCounter(&Time1);
		//auto Out = Embedding.Gather(Indices, 0, Thp);
		//Embedding.Assign(Embedding);
		//Ten114514.Permute({ 3,1,2,0 }).Clone();
		//DragonianLib::Tensor::Pad(Ten114514, {DragonianLib::None,19 },DragonianLib::PaddingType::Zero, DragonianLib::TensorType::Float32,nullptr, &Thp);
		/*DragonianLib::Tensor::Pad(
			Ten1919810,
			{DragonianLib::None,1 },
			DragonianLib::PaddingType::Replicate,
			DragonianLib::TensorType::Float32,
			nullptr, &Thp
		);*/
		//auto a = DragonianLib::Tensor::Diff(Ten1919810, 1, &Thp);
		//DragonianLib::Tensor::Stack({Ten1919810.Squeeze()}, 0, &Thp);
		//auto a = Ten1919810.Permute({ 0,2,1 });
		//DragonianLib::Tensor::Repeat(Ten1919810, { {0, 2} }, &Thp);
		//a.Continuous(&Thp);
		auto Res = ((Ten1919810 + Ten1919810) == Ten1919810 * 2.);
		std::cout << (bool)*(Res.Buffer()) << '\n';
		std::cout << (bool)*(Res.Buffer() + 1) << '\n';
		std::cout << (bool)*(Res.Buffer() + 2) << '\n';
		std::cout << (bool)*(Res.Buffer() + 3) << '\n';
		std::cout << (bool)*(Res.Buffer() + 4) << '\n';

		//Thp.Commit([&]() { a.Slice({ DragonianLib::None,{0,192} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ DragonianLib::None,{192,384} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ DragonianLib::None,{384,572} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ DragonianLib::None,{572,768} }).Continuous(); });
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
}

int main()
{
	{
		int OutputSamplingRate = 48000;
		int Channels = 1;
		bool OutFloat = true;
		bool OutPlanar = false;

		auto Audio = DragonianLib::AvCodec().Decode(
		   R"(D:/VSGIT/MoeSS - Release/Testdata/a.wav)",
		   OutputSamplingRate,
		   Channels,
		   OutFloat,
		   OutPlanar
		);

		auto AudioFloat = DragonianLib::AvCodec().DecodeFloat(
			R"(D:/VSGIT/MoeSS - Release/Testdata/a.wav)",
			OutputSamplingRate,
			Channels,
			OutPlanar
		);

		auto AudioSigned16 = DragonianLib::AvCodec().DecodeSigned16(
			R"(D:/VSGIT/MoeSS - Release/Testdata/a.wav)",
			OutputSamplingRate,
			Channels,
			OutPlanar
		);

		DragonianLib::WritePCMData(
			LR"(D:/VSGIT/MoeSS - Release/Testdata/testAudioSigned16.wav)",
			AudioSigned16,
			OutputSamplingRate,
			Channels,
			OutPlanar
		);

		DragonianLib::WritePCMData(
			LR"(D:/VSGIT/MoeSS - Release/Testdata/testAudioFloat.wav)",
			AudioFloat,
			OutputSamplingRate,
			Channels,
			OutPlanar
		);

		DragonianLib::WritePCMData(
			LR"(D:/VSGIT/MoeSS - Release/Testdata/testAudio.wav)",
			Audio,
			OutputSamplingRate,
			Channels,
			OutFloat,
			OutPlanar
		);
	}

	system("pause");
	//return 0;
	Demo();
	system("pause");
	return 0;
}