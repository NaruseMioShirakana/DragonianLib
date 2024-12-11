import torch
import time

emb = torch.nn.Embedding(100, 2048)
indices = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
indices = indices.repeat(1, 100)
ten = torch.ones(11451400)
for i in range(20):
    begin = time.time()
    torch.nn.functional.pad(ten, (114, 114))
    print(time.time() - begin)


'''
<Type Name="DragonianLib::Tensor&lt;*,*&gt;">
	<DisplayString>
		Shape: {_MyShape}, Data: {_MyData}
	</DisplayString>
	<Expand>
        <Synthetic Name="[ViewAttribute]">
			<DisplayString>
                {{ ViewShape={_MyShape}, ViewStep={_MyViewStep}, ViewLeftRange={_MyViewLeft}, ViewStride={_MyViewStride} }}
            </DisplayString>
			<Expand>
				<Item Name="[ViewShape]" ExcludeView="simple">_MyShape</Item>
				<Item Name="[ViewStep]" ExcludeView="simple">_MyViewStep</Item>
				<Item Name="[ViewLeft]" ExcludeView="simple">_MyViewLeft</Item>
				<Item Name="[ViewStride]" ExcludeView="simple">_MyViewStride</Item>
			</Expand>
		</Synthetic>


		<Item Name="Allocator">_MyAllocator</Item>
		<Item Name="First Element">_MyFirst</Item>
		<Item Name="Last Element">_MyLast</Item>
		<Item Name="Data">_MyData</Item>
		<Item Name="Shape">_MyShape</Item>
		<Item Name="View Step">_MyViewStep</Item>
		<Item Name="View Left">_MyViewLeft</Item>
		<Item Name="View Stride">_MyViewStride</Item>
		<Item Name="Is BroadCasted">IsBroadCasted_</Item>
	</Expand>
</Type>

<Type Name="DragonianLib::Tensor&lt;*&gt;">
	<DisplayString>{{ Type={ float }, Shape={ _MyShape }, Device={ _MyAllocator->Type_ } }}</DisplayString>
	<Expand>
		<Synthetic Name="[ViewAttribute]">
			<DisplayString>{{ ViewShape={ _MyShape }, ViewStep={ _MyViewStep }, ViewLeftRange={ _MyViewLeft }, ViewStride={ _MyViewStride } }}</DisplayString>
			<Expand>
				<Item Name="[ViewShape]" ExcludeView="simple">_MyShape</Item>
				<Item Name="[ViewStep]" ExcludeView="simple">_MyViewStep</Item>
				<Item Name="[ViewLeft]" ExcludeView="simple">_MyViewLeft</Item>
				<Item Name="[ViewStride]" ExcludeView="simple">_MyViewStride</Item>
			</Expand>
		</Synthetic>
		<Synthetic Name="[Data]">
			<DisplayString Condition="(bool)_MyData">{{ Shape = { _MyShape } }}</DisplayString>
			<DisplayString Condition="!(bool)_MyData">{{ This = { _MyData } }}</DisplayString>
			<Expand>
				<Synthetic Name="[Type]">
					<DisplayString Condition="(bool)_MyData">{{ Type = { float } }}</DisplayString>
				</Synthetic>
				<Item Name="[Buffer]" Condition="!(bool)_MyData">_MyFirst</Item>
			</Expand>
		</Synthetic>
	</Expand>
</Type>
'''