import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json


# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# Model architecture

def init_nerf_model_dynamic(D=8, W=256, input_ch=3, input_time=5, input_ch_views=3, output_ch=4, skips=[4,10], use_viewdirs=False):
    relu = tf.keras.layers.ReLU()

    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)
    input_time = int(input_time)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views + input_time))
    inputs_pts, inputs_views, inputs_time = tf.split(inputs, [input_ch, input_ch_views, input_time], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])
    inputs_time.set_shape([None, input_time])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape, inputs_time.shape)
    outputs = tf.concat([inputs_pts, inputs_time], -1)
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, inputs_time, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        outputs = tf.concat([bottleneck, inputs_views], -1)
        for i in range(1):
            outputs = dense(W // 2)(outputs)

        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def init_nerf_model(feature_vector=256, D=8, W=256, input_ch=3, input_time=None, input_ch_views=3, output_ch=4,
                    skips=[4, 10], use_viewdirs=False):
    relu = tf.keras.layers.ReLU()

    def dense(W, act=relu):
        return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))


    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts

    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)


    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)

        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.

        for i in range(1):
            outputs = dense(W // 2)(outputs)

        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model




def init_nerf_model_colorize(feature_vector=256, D=8, W=256, input_ch=3, input_time=None, input_ch_views=3, output_ch=4, skips=[4, 10], use_viewdirs=False):

    if input_time!=None:
        return init_nerf_model_dynamic(D=D, W=W, input_ch=input_ch, input_time=input_time,
                                       input_ch_views=input_ch_views, output_ch=output_ch,
                                       skips=skips, use_viewdirs=use_viewdirs)
    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))

    inputs2_coord = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs2_feature = tf.keras.Input(shape=feature_vector)
    feature_inputs = tf.keras.Input(shape=feature_vector)
    alpha_in = tf.keras.Input(shape=1)

    alpha_input = alpha_in

    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts_2, inputs_views_2 = tf.split(inputs2_coord, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_pts_2.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])
    inputs_views_2.set_shape([None, input_ch_views])
    inputs2_feature.set_shape([None, feature_vector])
    alpha_input.set_shape([None, 1])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape, inputs2_coord.shape, inputs_pts_2.shape, inputs_views_2.shape)
    outputs = inputs_pts
    alpha_input = dense(feature_vector//4)(alpha_input)
    inputs_pts_2 = dense(feature_vector)(tf.concat([inputs_pts_2, alpha_input], -1))
    outputs2 = tf.concat([inputs_pts_2, inputs2_feature], -1)

    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)
    for i in range(D):
        outputs2 = dense(W)(outputs2)
        if i in skips:
            outputs2 = tf.concat([inputs_pts_2, outputs2], -1)
    
    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        alpha_out_2 = dense(1, act=None)(outputs2)
        bottleneck = dense(256, act=None)(outputs)


        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        outputs2 = tf.concat([outputs2, inputs_views_2], -1)
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.

        # if this is the reference frame, return the 256 dimensional feature vector concat with view directions
        #if reference_frame=='True':
        #    print("output")
        #    return outputs
        # if a reference frame is provided, compute similarity matrix
        #else:

        
        for i in range(1):
            outputs = dense(W//2)(outputs)
        for i in range(1):
            outputs2 = dense(W//2)(outputs2)
        
        outputs = dense(3, act=None)(outputs)
        outputs2 = dense(3, act=None)(outputs2)
        outputs = tf.concat([outputs, alpha_out], -1)
        outputs2 = tf.concat([outputs2, alpha_out_2], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)
        outputs2 = dense(output_ch, act=None)(outputs2)

    outputs3 = feature_inputs
    for i in range(D):
        outputs3 = dense(feature_vector)(outputs3)
        if i in skips:
            outputs3 = tf.concat([feature_inputs, outputs3], -1)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, bottleneck, alpha_out])
    model2 = tf.keras.Model(inputs=[inputs2_coord, inputs2_feature, alpha_in], outputs=outputs2)
    model3 = tf.keras.Model(inputs=feature_inputs, outputs=outputs3)
    return model, model2, model3


# Ray helpers

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    #rays_d = np.dot(rays_d[... , :3],[0.114, 0.587, 0.299])
    #rays_o = np.dot(rays_o[... , :3],[0.114, 0.587, 0.299])
    #rays_d = rays_d.reshape(rays_d.shape[0], rays_d.shape[1],1)
    #rays_o = rays_o.reshape(rays_o.shape[0], rays_o.shape[1],1)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples
