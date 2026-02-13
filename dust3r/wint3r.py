# --------------------------------------------------------
# modified from CUT3R

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    transpose_to_landscape,
)
from layers.camera_head import CameraHead
from layers.depth_head import PtsHead
from layers.geometry import compute_relative_poses
from layers.pose_enc import pose_encoding_to_extri
from dust3r.patch_embed import get_patch_embed
from croco.models_croco.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import DecoderBlock, GlobalLocalDecoderBlock
from accelerate.logging import get_logger

printer = get_logger(__name__, log_level="DEBUG")

inf = float("inf")

def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)

class WinT3R(CroCoNet):

    def __init__(self,
                head_type="conv",  # or dpt
                freeze="none",
                landscape_only=False,
                patch_embed_cls="PatchEmbedDust3R",
                state_size=768,
                state_pe="2d",
                state_dec_num_heads=16,
                ckpts=None,
                pretrain_ckpts=None,
                window_size=4,
                **croco_kwargs,
                 ):
        self.gradient_checkpointing = True
        self.fixed_input_length = True
        croco_kwargs = fill_default_args(
            croco_kwargs, CrocoConfig.__init__
        )

        self.patch_embed_cls = patch_embed_cls
        self.croco_args = croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)

        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.register_tokens = nn.Embedding(state_size, self.enc_embed_dim)
        self.state_size = state_size
        self.state_pe = state_pe
        self.window_size = window_size
        self.cam_token = nn.Parameter(torch.randn(1, 1, self.dec_embed_dim))

        self.cam_head = CameraHead(dim_in=self.dec_embed_dim*2, pose_encoding_type="absT_quaR", window_size=self.window_size)
        nn.init.normal_(self.cam_token, std=1e-6)
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            head_type,
            landscape_only,
            **self.croco_args,
        )
        self.set_freeze(freeze)

        if pretrain_ckpts is not None:
            checkpoint = torch.load(ckpts, weights_only=False, map_location='cpu')
            res = self.load_pretrain(checkpoint, strict=False)

        if ckpts is not None:
            weights = torch.load(ckpts, weights_only=False)
            res = self.load_state_dict(weights, strict=False)
            print(f'Load checkpoints from {ckpts}: {res}')

    def load_pretrain(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        ckpt = ckpt['model']
        new_ckpt = dict(ckpt)
        # new_ckpt = new_ckpt['model']
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                GlobalLocalDecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.pts_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.dec_blocks,
                self.dec_blocks_state,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        head_type,
        landscape_only,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.head_type = head_type

        if head_type == "conv":
            self.pts_head = PtsHead(
                dim_in=self.dec_embed_dim,
                dim_out=[4],
            )
        else:
            raise NotImplementedError
        
        self.head = transpose_to_landscape(
            self.pts_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        )
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)

        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]

        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)

        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)

        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)

        selected_imgs = imgs
        selected_shapes = shapes
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i] += img_out[i]
        full_pos += img_pos

        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],       # [B, N, C]*n_views
            full_pos.chunk(len(views), dim=0),  # [B, N, C]*n_views
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim

        B, S, P, C = f_img.shape
        f_img = f_img.reshape(B, S*P, C)
        pos_img = pos_img.reshape(B, S*P, -1)
        final_output.append((f_state, f_img))

        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _ = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    use_reentrant=not self.fixed_input_length,
                )
                
                f_img, _, f_img_local= checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    [B, S, P, C],
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img)
                f_img, _, f_img_local= blk_img(*final_output[-1][::-1], pos_img, pos_state, [B, S, P, C])
            
            final_output.append((f_state, f_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return *zip(*final_output), f_img_local

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
    ):
        new_state_feat, dec, f_img_local = self._decoder(
            state_feat, state_pos, current_feat, current_pos
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, f_img_local

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        # mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        # init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            # init_mem,
            state_feat,
            state_pos,
            # mem,
        )

    def _forward_impl(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        bs = pos[0].shape[0]
        state_feat, state_pos = self._init_state(feat[0], pos[0])

        camera_pos = torch.zeros(bs, self.window_size, 1, pos[0].shape[2]).to(state_pos)
        camera_embed_list = []
        dec_list = []
        f_img_local_list = []
        view_idx_list = []
        # B, P, C = feat[0].shape
        for i in range(0, len(views), self.window_size//2):
            if i + self.window_size > len(views):
                break

            end_idx = i + self.window_size
            for j in range(i, end_idx):
                view_idx_list.append(j)
            feat_i = torch.stack(feat[i:end_idx], dim=1)
            feat_i = self.decoder_embed(feat_i)
            # B, S, P, C = feat_i.shape
            pos_i = torch.stack(pos[i:end_idx], dim=1) + 1

            cam_token = torch.stack([self.cam_token]*self.window_size, dim=1) 
           
            cam_token = cam_token.expand(bs, *cam_token.shape[1:])

            feat_i = torch.cat([cam_token, feat_i], dim=2)   # B, S, P, C

            B, S, P, C = feat_i.shape
            pos_i = torch.cat([camera_pos, pos_i], dim=2)
            new_state_feat, dec, f_img_local = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
            )

            assert len(dec) == self.dec_depth + 1

            dec = dec[-1].reshape(B, S, P, C)
            f_img_local = f_img_local.reshape(B, S, P, C)

            f_img_local_list.append(f_img_local[:, :, 1:].float())
            dec_list.append(dec[:, :, 1:].float())

            state_feat = new_state_feat
            
            camera_token = torch.cat([dec[:, :, 0], f_img_local[:, :, 0]], dim=-1)

            camera_embed_list.append(camera_token)

        camera_token = torch.cat(camera_embed_list, dim=1)

        head_input = [
            torch.cat(f_img_local_list, dim=1).reshape(-1, P-1, C).float(),
            torch.cat(dec_list, dim=1).reshape(-1, P-1, C).float(),
        ]

        with torch.amp.autocast(device_type='cuda', enabled=False):
            res = self.pts_head(head_input, views[0]["img"])
            camera_pos_encs = self.cam_head(camera_token)

        res['pts_local'] = res['pts_local'].reshape(B, len(view_idx_list), *res['pts_local'].shape[1:])   #B, S, H, W, 3
        res['conf'] = res['conf'].reshape(B, len(view_idx_list), *res['conf'].shape[1:])   #B, S, H, W
        res["camera_pos_enc"] = camera_pos_encs
        res['view_idx_list'] = view_idx_list

        return res
    
    def offline_inference(self, views, ret_first_pred=False):

        views_idxs = [[] for i in range(len(views))]

        for i in range(len(views)):
            views[i]["view_idx"] = i

        if len(views) < self.window_size:
            views.extend([views[-1]]*(self.window_size-len(views)))
        elif len(views)%(self.window_size//2)!=0:
            append_length = len(views)%(self.window_size//2)
            views.extend([views[-1]]*append_length)

        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        bs = pos[0].shape[0]
        state_feat, state_pos = self._init_state(feat[0], pos[0])

        camera_pos = torch.zeros(bs, self.window_size, 1, pos[0].shape[2]).to(state_pos)

        camera_embed_list = []
        dec_list = []
        f_img_local_list = []
        view_idx_list = []
        # B, P, C = feat[0].shape
        window_idx = -1
        for i in range(0, len(views), self.window_size//2):
            window_idx+=1
            if i + self.window_size > len(views):
                break

            end_idx = i + self.window_size
            for j in range(0, self.window_size):
                view_idx_list.append(i+j)
                views_idxs[views[i+j]["view_idx"]].append(window_idx*self.window_size+j)
            feat_i = torch.stack(feat[i:end_idx], dim=1)
            feat_i = self.decoder_embed(feat_i)
            # B, S, P, C = feat_i.shape
            pos_i = torch.stack(pos[i:end_idx], dim=1) + 1

            cam_token = torch.stack([self.cam_token]*self.window_size, dim=1) 
           
            cam_token = cam_token.expand(bs, *cam_token.shape[1:])

            feat_i = torch.cat([cam_token, feat_i], dim=2)   # B, S, P, C

            B, S, P, C = feat_i.shape
            pos_i = torch.cat([camera_pos, pos_i], dim=2)
            new_state_feat, dec, f_img_local = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
            )

            assert len(dec) == self.dec_depth + 1
            dec = dec[-1].reshape(B, S, P, C)
            f_img_local = f_img_local.reshape(B, S, P, C)
            f_img_local_list.append(f_img_local[:, :, 1:].float())
            dec_list.append(dec[:, :, 1:].float())

            state_feat = new_state_feat
            camera_token = torch.cat([dec[:, :, 0], f_img_local[:, :, 0]], dim=-1)

            camera_embed_list.append(camera_token)

        camera_token = torch.cat(camera_embed_list, dim=1).float()

        head_input = [
            torch.cat(f_img_local_list, dim=1).reshape(-1, P-1, C).float(),
            torch.cat(dec_list, dim=1).reshape(-1, P-1, C).float(),
        ]

        with torch.amp.autocast(device_type='cuda', enabled=False):
            res = self.pts_head(head_input, views[0]["img"])
            camera_pos_encs = self.cam_head(camera_token)

        res['pts_local'] = res['pts_local'].reshape(B, len(view_idx_list), *res['pts_local'].shape[1:])   #B, S, H, W, 3
        res['conf'] = res['conf'].reshape(B, len(view_idx_list), *res['conf'].shape[1:])   #B, S, H, W
        res["camera_pos_enc"] = camera_pos_encs

        final_ress = []
        for views_ind in views_idxs:
            conf_score = res['conf'][:, views_ind].sum(dim=[-1, -2])
            max_indices_in_selected = torch.argmax(conf_score, dim=1)
            idx_tensor = torch.tensor(views_ind,device=conf_score.device)
            original_indices = idx_tensor[max_indices_in_selected]
            final_ress.append(original_indices)

        final_ress = torch.stack(final_ress, dim=1)
        res['pts_local'] = torch.gather(res['pts_local'], dim=1, index=final_ress.view(*final_ress.shape,*([1]*(res['pts_local'].ndim-final_ress.ndim))).expand(-1, -1, *res['pts_local'].shape[2:]))
        res['conf'] = torch.gather(res['conf'], dim=1, index=final_ress.view(*final_ress.shape,*([1]*(res['conf'].ndim-final_ress.ndim))).expand(-1, -1, *res['conf'].shape[2:]))

        if ret_first_pred:
            views_idxs = [min(idxs) for idxs in views_idxs]
            # res['conf'] = res['conf'][:, views_idxs]
            res["camera_pos_enc"] = [cam_pose_enc[:, views_idxs] for cam_pose_enc in res["camera_pos_enc"]]
        else:
            views_idxs = [max(idxs) for idxs in views_idxs]
            # res['conf'] = res['conf'][:, views_idxs]
            res["camera_pos_enc"] = [cam_pose_enc[:, views_idxs] for cam_pose_enc in res["camera_pos_enc"]]

        # Translate the relative coordinate system to the coordinate system relative to the first frame.
        res["camera_pos_enc"] = [compute_relative_poses(camera_pos_en) for camera_pos_en in res["camera_pos_enc"]]

        extrinsics = pose_encoding_to_extri(res["camera_pos_enc"][-1])
        R_cam_to_world = extrinsics[:, :, :3, :3]     #  B, S, 3, 3
        t_cam_to_world = extrinsics[:, :, :3, 3]      #  B, S, 3
        world_coords_points = torch.einsum("bsij,bshwj->bshwi", R_cam_to_world, res['pts_local']) + t_cam_to_world[:, :, None, None]   #B, S, H, W, 3
        res['pts3d_in_other_view'] = world_coords_points


        return res
    
    def online_inference(self, views, ret_first_pred=False):

        views_idxs = [[] for i in range(len(views))]

        for i in range(len(views)):
            views[i]["view_idx"] = i

        if len(views) < self.window_size:
            views.extend([views[-1]]*(self.window_size-len(views)))
        elif len(views)%(self.window_size//2)!=0:
            append_length = len(views)%(self.window_size//2)
            views.extend([views[-1]]*append_length)

        state_feat, state_pos = None, None

        camera_embed_list = []
        view_idx_list = []
        # B, P, C = feat[0].shape
        window_idx = -1
        res = {}
        for i in range(0, len(views), self.window_size//2):
            window_idx+=1
            if i + self.window_size > len(views):
                break

            end_idx = i + self.window_size
            for j in range(0, self.window_size):
                view_idx_list.append(i+j)
                views_idxs[views[i+j]["view_idx"]].append(window_idx*self.window_size+j)
            
            shape, feat_ls, pos = self._encode_views(views[i:end_idx])
            feat = feat_ls[-1]
            if state_feat is None:
                bs = pos[0].shape[0]
                state_feat, state_pos = self._init_state(feat[0], pos[0])
                camera_pos = torch.zeros(bs, self.window_size, 1, pos[0].shape[2]).to(state_pos)

            feat_i = torch.stack(feat, dim=1)
            feat_i = self.decoder_embed(feat_i)

            pos_i = torch.stack(pos, dim=1) + 1

            cam_token = torch.stack([self.cam_token]*self.window_size, dim=1) 
           
            cam_token = cam_token.expand(bs, *cam_token.shape[1:])

            feat_i = torch.cat([cam_token, feat_i], dim=2)   # B, S, P, C

            B, S, P, C = feat_i.shape
            pos_i = torch.cat([camera_pos, pos_i], dim=2)
            new_state_feat, dec, f_img_local = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
            )

            assert len(dec) == self.dec_depth + 1
            dec = dec[-1].reshape(B, S, P, C)
            f_img_local = f_img_local.reshape(B, S, P, C)
            state_feat = new_state_feat
            camera_token = torch.cat([dec[:, :, 0], f_img_local[:, :, 0]], dim=-1)
            camera_embed_list.append(camera_token)
            camera_token = torch.cat(camera_embed_list, dim=1).float()
            head_input = [
                f_img_local[:, :, 1:].float().reshape(-1, P-1, C).float(),
                dec[:, :, 1:].float().reshape(-1, P-1, C).float(),
            ]
            camera_token = torch.cat(camera_embed_list, dim=1).float()

            with torch.amp.autocast(device_type='cuda', enabled=False):
                ress = self.pts_head(head_input, views[0]["img"])
                camera_pos_encs = self.cam_head(camera_token)
                res["camera_pos_enc"] = camera_pos_encs
                if 'pts_local' not in res.keys():
                    res['pts_local'] = ress['pts_local'].reshape(B, S, *ress['pts_local'].shape[1:])
                    res['conf'] = ress['conf'].reshape(B, S, *ress['conf'].shape[1:])
                else:
                    res['pts_local'] = torch.cat([res['pts_local'], ress['pts_local'].reshape(B, S, *ress['pts_local'].shape[1:])], dim=1)
                    res['conf'] = torch.cat([res['conf'], ress['conf'].reshape(B, S, *ress['conf'].shape[1:])], dim=1)

        final_ress = []
        for views_ind in views_idxs:
            conf_score = res['conf'][:, views_ind].sum(dim=[-1, -2])
            max_indices_in_selected = torch.argmax(conf_score, dim=1)
            idx_tensor = torch.tensor(views_ind,device=conf_score.device)
            original_indices = idx_tensor[max_indices_in_selected]
            final_ress.append(original_indices)

        final_ress = torch.stack(final_ress, dim=1)
        res['pts_local'] = torch.gather(res['pts_local'], dim=1, index=final_ress.view(*final_ress.shape,*([1]*(res['pts_local'].ndim-final_ress.ndim))).expand(-1, -1, *res['pts_local'].shape[2:]))
        res['conf'] = torch.gather(res['conf'], dim=1, index=final_ress.view(*final_ress.shape,*([1]*(res['conf'].ndim-final_ress.ndim))).expand(-1, -1, *res['conf'].shape[2:]))

        if ret_first_pred:
            views_idxs = [min(idxs) for idxs in views_idxs]
            # res['conf'] = res['conf'][:, views_idxs]
            res["camera_pos_enc"] = [cam_pose_enc[:, views_idxs] for cam_pose_enc in res["camera_pos_enc"]]
        else:
            views_idxs = [max(idxs) for idxs in views_idxs]
            # res['conf'] = res['conf'][:, views_idxs]
            res["camera_pos_enc"] = [cam_pose_enc[:, views_idxs] for cam_pose_enc in res["camera_pos_enc"]]

        # Translate the relative coordinate system to the coordinate system relative to the first frame.
        res["camera_pos_enc"] = [compute_relative_poses(camera_pos_en) for camera_pos_en in res["camera_pos_enc"]]

        extrinsics = pose_encoding_to_extri(res["camera_pos_enc"][-1])
        R_cam_to_world = extrinsics[:, :, :3, :3]     #  B, S, 3, 3
        t_cam_to_world = extrinsics[:, :, :3, 3]      #  B, S, 3
        world_coords_points = torch.einsum("bsij,bshwj->bshwi", R_cam_to_world, res['pts_local']) + t_cam_to_world[:, :, None, None]   #B, S, H, W, 3
        res['pts3d_in_other_view'] = world_coords_points

        return res

    def forward(self, views, ret_first_pred=False, mode="train"):
        if mode == 'train':
            ress = self._forward_impl(views)
        elif mode == "online":
            ress = self.online_inference(views, ret_first_pred=ret_first_pred)
        elif mode == 'offline':
            ress = self.offline_inference(views, ret_first_pred=ret_first_pred)

        return ress

if __name__ == "__main__":
    print(WinT3R.mro())
