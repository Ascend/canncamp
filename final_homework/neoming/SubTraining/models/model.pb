
=
PlaceholderPlaceholder*
shape:	?*
dtype0
c

weights_l1Const*
dtype0*A
value8B6
"(?G?ߢپ??mUm@GJ?@?&??A@4???c?UA???
O
weights_l1/readIdentity
weights_l1*
T0*
_class
loc:@weights_l1
`
bias_l1Const*
dtype0*A
value8B6
"(?~?G????3ͿH????d?U/??S?!???-7??H???
F
bias_l1/readIdentitybias_l1*
T0*
_class
loc:@bias_l1
]
MatMulMatMulPlaceholderweights_l1/read*
transpose_b( *
T0*
transpose_a( 
+
addAddV2MatMulbias_l1/read*
T0
*
output_l1_sigmoidSigmoidadd*
T0
c

weights_l2Const*
dtype0*A
value8B6
"(ry?	???U?@?\?l3u@?@a??>ԭ?>?????
O
weights_l2/readIdentity
weights_l2*
T0*
_class
loc:@weights_l2
<
bias_l2Const*
dtype0*
valueB*<)??
F
bias_l2/readIdentitybias_l2*
T0*
_class
loc:@bias_l2
e
MatMul_1MatMuloutput_l1_sigmoidweights_l2/read*
transpose_b( *
T0*
transpose_a( 
/
add_1AddV2MatMul_1bias_l2/read*
T0
%

predictionSigmoidadd_1*
T0 