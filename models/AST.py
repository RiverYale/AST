import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vit import vit_small_patch16_224

class AST(nn.Module):
	def __init__(self, args):
		super(AST, self).__init__()

		self.drone_module = vit_small_patch16_224(args, (args.drone_h, args.drone_w), pretrained=True)
		if args.share_weight:
			self.sate_module = self.drone_module
		else:
			self.sate_module = vit_small_patch16_224(args, (args.sate_h, args.sate_w), pretrained=True)

	def forward(self, drone_img, sate_img, paths=None):
		drone_path = paths[0] if paths is not None else None
		sate_path = paths[1] if paths is not None else None

		sate_res = self.sate_module(sate_img, path=sate_path)
		drone_res = self.drone_module(drone_img, path=drone_path)

		return drone_res, sate_res

	def reshape_pos_embed(self, old_grid_size, new_grid_size, num_prefix_tokens, posemb):
		posemb_prefix = posemb[:, :num_prefix_tokens]
		posemb_grid = posemb[:, num_prefix_tokens:]
		posemb_grid = posemb_grid.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute((0, 3, 1, 2))
		posemb_grid = F.interpolate(posemb_grid, size=new_grid_size, mode='bilinear', align_corners=False)	# bicubic / bilinear
		posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, new_grid_size[0] * new_grid_size[1], -1)
		posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
		return posemb

	def load_checkpoint(self, checkpoint):
		num_prefix_tokens = self.drone_module.num_prefix_tokens
		old_grid_size = int(math.sqrt(checkpoint['drone_module.pos_embed'].shape[1]-num_prefix_tokens))
		old_grid_size = (old_grid_size, old_grid_size)
		new_grid_size = self.drone_module.patch_embed.grid_size
		if old_grid_size != new_grid_size:
			reshaped_pos_embed = self.reshape_pos_embed(old_grid_size, new_grid_size, num_prefix_tokens, checkpoint['drone_module.pos_embed'])
			checkpoint['drone_module.pos_embed'] = reshaped_pos_embed

		num_prefix_tokens = self.sate_module.num_prefix_tokens
		old_grid_size = int(math.sqrt(checkpoint['sate_module.pos_embed'].shape[1]-num_prefix_tokens))
		old_grid_size = (old_grid_size, old_grid_size)
		new_grid_size = self.sate_module.patch_embed.grid_size
		if old_grid_size != new_grid_size:
			reshaped_pos_embed = self.reshape_pos_embed(old_grid_size, new_grid_size, num_prefix_tokens, checkpoint['sate_module.pos_embed'])
			checkpoint['sate_module.pos_embed'] = reshaped_pos_embed

		self.load_state_dict(checkpoint)
