import numpy as np

def separable_conv2d(C, H, W, R_H, R_W, F, input_t, 
                               dweights, pweights):
    """ Depthwise separable convolution.
    C: number of input channels
    H: spatial height of input
    W: spatial width of input
    R_H: conv kernel height
    R_W: conv kernel width
    F: number of features (kernels)
    input_t[C, H, W]: input activations
    dweights[C, R_H, R_W]: depth weights
    pweights[F, C]: point weights
    output_t[F, H, W]: output
    """

    output_t = np.zeros([F, H, W])

    # same padding (channel first)
    p_w = R_W // 2
    p_h = R_H // 2
    pad_width = ((0, 0), (p_h, p_h), (p_w, p_w))
    pad_input = np.pad(input_t, pad_width=pad_width,
                                    mode='constant',
                                  constant_values=0)

    # depthwise convolution
    depthwise_output = np.zeros((C, H, W))
    for c in range(C):
        for i in range(H):
            for j in range(W):
                for fi in range(R_H):
                    for fj in range(R_W):
                        w = dweights[c, fi, fj]
                        depthwise_output[c, i, j] += pad_input[c, i + fi, j + fj] * w
    
    # pointwise convolution
    for out_c in range(F):
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    w = pweights[out_c, c]
                    output_t[out_c, i, j] += depthwise_output[c, i, j] * w

    return output_t