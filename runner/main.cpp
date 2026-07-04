#include <iostream>
#include <cstdio>
#include <memory>

#include "tensorflow/lite/version.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

int main(int argc, char* argv[]) {
	std::cout << "TensorFlow Lite version: " << TFLITE_VERSION_STRING << std::endl;
	std::cout << "TensorFlow Lite schema version: " << TFLITE_SCHEMA_VERSION << std::endl;

    if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <model.tflite>" << std::endl;
        return 1;
    }
    const char* model_path = argv[1];

    // 1. Load the model from file
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
		std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    // 2. Build the interpreter with the built-in op resolver
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
		std::cerr << "Failed to construct interpreter" << std::endl;
        return 1;
    }

    // Print a summary of the model's I/O
    tflite::PrintInterpreterState(interpreter.get());

    // 3. Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk) {
		std::cerr << "Failed to allocate tensors" << std::endl;
        return 1;
    }

    // 4. Fill input tensor(s) with data
    // Assumes a single float32 input tensor
    float* input = interpreter->typed_input_tensor<float>(0);
    TfLiteTensor* input_tensor = interpreter->input_tensor(0);
    int input_size = 1;
    for (int i = 0; i < input_tensor->dims->size; ++i) {
        input_size *= input_tensor->dims->data[i];
    }
    for (int i = 0; i < input_size; ++i) {
        input[i] = 0.0f;  // replace with real data
    }

    // 5. Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    // 6. Read output tensor(s)
    float* output = interpreter->typed_output_tensor<float>(0);
    TfLiteTensor* output_tensor = interpreter->output_tensor(0);
    int output_size = 1;
    for (int i = 0; i < output_tensor->dims->size; ++i) {
        output_size *= output_tensor->dims->data[i];
    }
    for (int i = 0; i < output_size; ++i) {
		std::cout << "output[" << i << "] = " << output[i] << std::endl;
    }

    return 0;
}
