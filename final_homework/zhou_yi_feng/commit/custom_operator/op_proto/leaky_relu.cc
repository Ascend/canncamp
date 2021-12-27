#include "leaky_relu.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(LeakyReluInferShape)
{
    auto x_shape = op.GetInputDescByName("x").GetShape().GetDims();
    DataType x_dtype = op.GetInputDescByName("x").GetDataType();
    TensorDesc y_desc = op.GetOutputDescByName("y");
    y_desc.SetShape(ge::Shape(x_shape));
    y_desc.SetDataType(x_dtype);
    (void)op.UpdateOutputDesc("y", y_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(LeakyRelu, LeakyReluVerify)
{

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LeakyRelu, LeakyReluInferShape);
VERIFY_FUNC_REG(LeakyRelu, LeakyReluVerify);

}  // namespace ge
