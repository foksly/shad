def compute_conv_output_size(image_size, padding, kernel_size, stride, dilation=1):
    '''https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d'''

    if isinstance(image_size, int):
        out_image_size = int((image_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        return out_image_size
    elif isinstance(image_size, tuple):
        h, w = image_size
        out_image_h = compute_conv_output_size(h, padding, kernel_size, stride, dilation)
        out_image_w = compute_conv_output_size(w, padding, kernel_size, stride, dilation)
        return (out_image_h, out_image_w)
