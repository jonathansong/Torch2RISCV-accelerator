// IREE patch 0005 (compiler/patches/iree/README.md): a chain of insert_slices stored
// whole folds into one store per slice. Without the patch IREE's
// FoldInsertSliceWithTensorStoreOp kept only the last insert (it dropped the
// destination), and the earlier slices' values were lost (fixed upstream by
// iree-org/iree#24266 and #24276, after the pinned revision).
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-tensor-to-flow))" %s | FileCheck %s

// CHECK-LABEL: util.func public @insert_chain
//       CHECK:   flow.dispatch.workgroups
//   CHECK-DAG:     iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1, 4]
//   CHECK-DAG:     iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [1, 0], sizes = [1, 4]
//   CHECK-NOT:     tensor.insert_slice
util.func public @insert_chain(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<2x4xf32> {
  %0 = flow.dispatch.workgroups(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32> =
      (%ia: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>, %ib: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>,
       %out: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x4xf32>>) {
    %va = iree_tensor_ext.dispatch.tensor.load %ia, offsets = [0], sizes = [4], strides = [1]
        : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
    %vb = iree_tensor_ext.dispatch.tensor.load %ib, offsets = [0], sizes = [4], strides = [1]
        : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
    %e = tensor.empty() : tensor<2x4xf32>
    %r0 = tensor.insert_slice %va into %e[0, 0] [1, 4] [1, 1] : tensor<4xf32> into tensor<2x4xf32>
    %r1 = tensor.insert_slice %vb into %r0[1, 0] [1, 4] [1, 1] : tensor<4xf32> into tensor<2x4xf32>
    iree_tensor_ext.dispatch.tensor.store %r1, %out, offsets = [0, 0], sizes = [2, 4], strides = [1, 1]
        : tensor<2x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x4xf32>>
    flow.return
  }
  util.return %0 : tensor<2x4xf32>
}
