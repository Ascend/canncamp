#include "sub.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(SubInferShape)
{
    auto x_shape = op.GetInputDescByName("x").GetShape().GetDims();
    DataType x_dtype = op.GetInputDescByName("x").GetDataType();
    TensorDesc z_desc = op.GetOutputDescByName("z");
    z_desc.SetShape(ge::Shape(x_shape));
    z_desc.SetDataType(x_dtype);
    (void)op.UpdateOutputDesc("z", z_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Sub, SubVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Sub, SubInferShape);
VERIFY_FUNC_REG(Sub, SubVerify);

}  // namespace ge
