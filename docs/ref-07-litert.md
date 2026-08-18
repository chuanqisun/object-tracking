LiteRT.js is Google's high performance WebAI runtime, targeting production Web
applications. It is a continuation of the LiteRT stack, ensuring
[multi-framework](https://developers.google.com/edge/conversion/overview) support and unifying our core runtime
across all platforms.

LiteRT.js supports the following core features:

1. **In-browser hardware-accelerated inference** : Run models with exceptional CPU performance, accelerated by [XNNPack](https://github.com/google/XNNPACK) mapped to lightweight WebAssembly (Wasm). For GPU and dedicated hardware scaling (such as NPUs), LiteRT.js natively surfaces both the [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) API and the emerging [WebNN](https://webmachinelearning.github.io/webnn/) API empowering fine grained platform-specific optimization.
2. **Multi-framework compatibility**: Streamline development semantics by compiling from your preferred ML Framework natively: PyTorch, JAX or TensorFlow.
3. **Iterate on existing pipelines**: Out-the-box integration with existing TensorFlow.js architectures by parsing natively supported TensorFlow.js Tensors as direct boundary inputs and outputs.

## Installation

Install the `@litertjs/core` package from npm:

    npm install @litertjs/core

The Wasm files are located in `node_modules/@litertjs/core/wasm/`. For
convenience, copy and serve the entire `wasm/` folder. Then, import the package
and load the Wasm files:

    import {loadLiteRt} from '@litertjs/core';

    // Load the LiteRT.js Wasm files from a CDN.
    await loadLiteRt('https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/')
    // Alternatively, host them from your server.
    // They are located in node_modules/@litertjs/core/wasm/
    await loadLiteRt(`your/path/to/wasm/`);

## Model conversion

LiteRT.js uses the same `.tflite` format as the rest of the LiteRT ecosystem,
and it supports existing models on
[Kaggle](https://www.kaggle.com/models?framework=tfLite) and
[Huggingface](https://huggingface.co/models?library=tflite&sort=trending). If
you have a new PyTorch model, you'll need to convert it.

### Convert a PyTorch Model to LiteRT

To convert a PyTorch model to LiteRT, use the [litert-torch
converter](https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md#conversion).

    import litert_torch

    # Load your torch model. We're using resnet for this example.
    resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    sample_inputs = (torch.randn(1, 3, 224, 224),)

    # Convert the model to LiteRT.
    edge_model = litert_torch.convert(resnet18.eval(), sample_inputs)

    # Export the model.
    edge_model.export('resnet.tflite')

### Run the Converted Model

After converting the model to a `.tflite` file, you can run it in the browser.

    import {loadAndCompile} from '@litertjs/core';

    // Load the model hosted from your server. This makes an http(s) request.
    const model = await loadAndCompile('/path/to/model.tflite', {
        accelerator: 'webgpu',
        // Can select from 'webnn', 'webgpu', & 'wasm'.
        // Unsupported ops on webgpu & webnn automatically fallback to CPU.
    });
    // The model can also be loaded from a Uint8Array if you want to fetch it yourself.

    // Create image input data
    const image = new Float32Array(224 * 224 * 3).fill(0);
    const inputTensor = new Tensor(image, /* shape */ [1, 3, 224, 224]);

    // Run the model
    const outputs = await model.run(inputTensor);
    // You can also use `await model.run([inputTensor]);`
    // or `await model.run({'input_tensor_name': inputTensor});`

    // Clean up and get outputs
    inputTensor.delete();
    const output = outputs[0];
    const outputData = await output.data();
    output.delete();

## Integrate into existing TensorFlow.js pipelines

You should consider integrating LiteRT.js into your
[TensorFlow.js](https://www.tensorflow.org/js) pipelines for the following
reasons:

1. **Exceptional GPU \& Hardware Performance**: LiteRT.js models leverage WebGPU acceleration for optimized performance across browser architectures. With support for WebGPU and upcoming WebNN, LiteRT.js offers flexible hardware acceleration across a variety of edge devices.
2. **Easier Model Conversion Path**: The LiteRT.js conversion path goes directly from PyTorch to LiteRT. The PyTorch to TensorFlow.js conversion path is significantly more complicated, requiring you to go from PyTorch -\> ONNX -\> TensorFlow -\> TensorFlow.js.
3. **Debugging tools** : The LiteRT.js conversion path comes with [debugging
   tools](https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md#debugging--reporting-errors).

LiteRT.js is designed to function within TensorFlow.js pipelines, and is
compatible with TensorFlow.js pre- and post-processing, so the only thing you
need to migrate is the model itself.

Integrate LiteRT.js into TensorFlow.js pipelines with the following steps:

1. Convert your original TensorFlow, JAX, or PyTorch model to `.tflite`. For details, see the [model conversion](https://developers.google.com/edge/litert/web#model-conversion) section.
2. Install the `@litertjs/core` and `@litertjs/tfjs-interop` NPM packages.
3. Import and use the [TensorFlow.js WebGPU
   backend](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgpu). This is required for LiteRT.js to interoperate with TensorFlow.js.
4. Replace [loading the TensorFlow.js
   model](https://js.tensorflow.org/api/latest/#loadGraphModel) with [loading
   the LiteRT.js model](https://developers.google.com/edge/litert/web#run-converted).
5. Substitute the TensorFlow.js [`model.predict`](https://js.tensorflow.org/api/latest/#tf.GraphModel.predict)(inputs) or [`model.execute`](https://js.tensorflow.org/api/latest/#tf.GraphModel.execute)(inputs) with `runWithTfjsTensors`(liteRtModel, inputs). `runWithTfjsTensors` takes the same input tensors that TensorFlow.js models use and outputs TensorFlow.js tensors.
6. Test that the model pipeline outputs the results you expect.

Using LiteRT.js with `runWithTfjsTensors` may also require the following changes
to the model inputs:

1. **Reorder inputs**: Depending on how the converter ordered the inputs and outputs of the model, you may need to change their order as you pass them in.
2. **Transpose inputs** : It's also possible that the converter changed the layout of the inputs and outputs of the model compared to what TensorFlow.js uses. You may need to [transpose](https://js.tensorflow.org/api/latest/#transpose) your inputs to match the model and outputs to match the rest of the pipeline.
3. **Rename inputs**: If you're using named inputs, the names may have also changed.

You can get more information about the inputs and outputs of the model with
`model.getInputDetails()` and `model.getOutputDetails()`.
