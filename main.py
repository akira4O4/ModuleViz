from modeuls.conv import Conv
import torch

if __name__ == '__main__':
    model = Conv(3, 32, 3)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        f"{str(model)}.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
