
#include "random_walk_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace std;
//length，代表元素个数，size代表字节数。
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  size_t usrSize = 256; // 设置用户需要使用的workspace大小为256字节。
  auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());
  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
  currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  RandomWalkCustomTilingData tiling;
  uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
  context->SetBlockDim(aivNum);
  // Get input and output size，unit is elements.
  const gert::StorageShape* seeds_data_shape = context->GetInputShape(0);
  const gert::StorageShape* row_ptr_shape = context->GetInputShape(1);
  const gert::StorageShape* col_ind_shape = context->GetInputShape(2);
  const gert::StorageShape* metapath_data_shape = context->GetInputShape(3);
  const gert::StorageShape* randomNumber_data_shape = context->GetInputShape(4);
  const gert::StorageShape* output_shape = context->GetOutputShape(0);

  ge::DataType tensor_dtype = context->GetInputDesc(0)->GetDataType();
  uint32_t dataType = 0;
  uint32_t dataSize = 0;
  if(tensor_dtype == ge::DT_INT32) {
    // printf("tensor_dtype is INT32\n");
    dataType = 1;
    dataSize = 4;
  }
   else if(tensor_dtype == ge::DT_INT64) {
    // printf("tensor_dtype is INT64\n");
    dataType = 2;
    dataSize = 8;
  } 
  tiling.set_dataType(dataType);
  tiling.set_dataSize(dataSize);
  // printf("set dataType and dataSize done\n");
  uint32_t elementAligned = 32 / tiling.get_dataSize();
  uint32_t seedsDataLength = 1;
  for (int i = 0; i < seeds_data_shape->GetStorageShape().GetDimNum(); i++)
    seedsDataLength *= seeds_data_shape->GetStorageShape().GetDim(i);
  uint32_t seedsDataLengthAligned = (seedsDataLength + (elementAligned - 1))/ elementAligned * elementAligned;
  uint32_t rowPtrLength = 1;
  for (int i = 0; i < row_ptr_shape->GetStorageShape().GetDimNum(); i++)
    rowPtrLength *= row_ptr_shape->GetStorageShape().GetDim(i);
  uint32_t rowPtrLengthAligned = (rowPtrLength + (elementAligned - 1))/ elementAligned * elementAligned;
  uint32_t colIndLength = 1;
  for (int i = 0; i < col_ind_shape->GetStorageShape().GetDimNum(); i++)
    colIndLength *= col_ind_shape->GetStorageShape().GetDim(i);
  uint32_t colIndLengthAligned = (colIndLength + (elementAligned - 1))/ elementAligned * elementAligned;
  uint32_t metapathDataLength = 1;
 for (int i = 0; i < metapath_data_shape->GetStorageShape().GetDimNum(); i++)
    metapathDataLength *= metapath_data_shape->GetStorageShape().GetDim(i);
  uint32_t metapathDataLengthAligned = (metapathDataLength + (elementAligned - 1))/ elementAligned * elementAligned;
  uint32_t randomNumberDataLength = 1;
  for (int i = 0; i < randomNumber_data_shape->GetStorageShape().GetDimNum(); i++)
    randomNumberDataLength *= randomNumber_data_shape->GetStorageShape().GetDim(i);
  uint32_t randomNumberDataLengthAligned = (randomNumberDataLength + (elementAligned - 1))/ elementAligned * elementAligned;
  uint32_t outputLength = 1;
  for (int i = 0; i < output_shape->GetStorageShape().GetDimNum(); i++)
    outputLength *= output_shape->GetStorageShape().GetDim(i);
  uint32_t outputLengthAligned = (outputLength + (elementAligned - 1))/ elementAligned * elementAligned;

  //计算unpad参数。
  vector<int64_t> shapeVec = {64,metapathDataLengthAligned};
  ge::Shape srcShape(shapeVec);
  uint32_t maxValue = 0;
  uint32_t minValue = 0;
  AscendC::GetUnPadMaxMinTmpSize(ascendcPlatform, srcShape, 4, maxValue, minValue);
  const uint32_t localWorkSpaceSize = minValue;
  AscendC::UnPadTilingFunc(srcShape, localWorkSpaceSize , 4, tiling.UnPadTilingData);
  //计算留给seeds_data的ub空间大小,这里先默认一次能够搬入所有的图数据。
  uint32_t ubLeftSize = ub_size - rowPtrLengthAligned * tiling.get_dataSize() - colIndLengthAligned * tiling.get_dataSize() - metapathDataLengthAligned * tiling.get_dataSize() - randomNumberDataLengthAligned * tiling.get_dataSize();
  if(ubLeftSize < seedsDataLengthAligned * tiling.get_dataSize()) {
    std::cout << "ubLeftSize is not enough for seeds_data" << std::endl;
    return ge::GRAPH_FAILED;
  }
  //计算分核,分块由算子的process函数负责。
  uint32_t baseBlocksPerCore = seedsDataLengthAligned * tiling.get_dataSize() / 32 / aivNum; // 小核每个核分几个block，一个block32字节
  tiling.set_seedsDataLength(seedsDataLength);
  tiling.set_seedsDataLengthAligned(seedsDataLengthAligned);
  tiling.set_rowPtrLength(rowPtrLength);
  tiling.set_rowPtrLengthAligned(rowPtrLengthAligned);
  tiling.set_colIndLength(colIndLength);
  tiling.set_colIndLengthAligned(colIndLengthAligned);
  tiling.set_metapathDataLength(metapathDataLength);
  tiling.set_metapathDataLengthAligned(metapathDataLengthAligned);
  tiling.set_randomNumberDataLength(randomNumberDataLength);
  tiling.set_randomNumberDataLengthAligned(randomNumberDataLengthAligned);
  tiling.set_outputLength(outputLength);
  tiling.set_outputLengthAligned(outputLengthAligned);
  tiling.set_ubLeftSize(ubLeftSize);
  tiling.set_baseBlocksPerCore(baseBlocksPerCore);

 

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  // printf("tiling Done!\n");
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* seeds_shape = context->GetInputShape(0);
    const gert::Shape* metapath_shape = context->GetInputShape(3);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = gert::Shape({seeds_shape->GetShapeSize() * (metapath_shape->GetShapeSize() + 1)});
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class RandomWalkCustom : public OpDef {
public:
    explicit RandomWalkCustom(const char* name) : OpDef(name)
    {
        this->Input("seeds_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("row_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("col_ind")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("metapath_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("randomNumber_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(RandomWalkCustom);
}
