import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.hub import download_cached_file
from timm.models.helpers import checkpoint_seq


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            # x = self.classifier(x)
            return x


class Vit(VisionTransformer):
	def __init__(self, part_num=0, ft_dim=512, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.part_num = part_num
		self.ft_dim = ft_dim

		num_classes = kwargs['num_classes']
		embed_dim = kwargs['embed_dim']
		self.global_head = ClassBlock(embed_dim, num_classes, 0.5, num_bottleneck=ft_dim)
		for i in range(part_num):
			name = f'part_head_{i}'
			attr = ClassBlock(embed_dim, num_classes, 0.5, num_bottleneck=ft_dim)
			setattr(self, name, attr)

	def forward_features_atten(self, x):
		x = self.patch_embed(x)
		x = self._pos_embed(x)
		if self.grad_checkpointing and not torch.jit.is_scripting():
			x = checkpoint_seq(self.blocks, x)
		else:
			# self.blocks(x)
			for i, blk in enumerate(self.blocks):
				if i == len(self.blocks)-1:
					y = blk.norm1(x)
					B, N, C = y.shape
					qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
					q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
					att = (q @ k.transpose(-2, -1)) * blk.attn.scale

					atten_map = att

					att = att.softmax(dim=-1)
					att = blk.attn.attn_drop(att)

					y = (att @ v).transpose(1, 2).reshape(B, N, C)
					y = blk.attn.proj(y)
					y = blk.attn.proj_drop(y)
					
					x = x + blk.drop_path1(blk.ls1(y))
					x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
				else:
					x = blk(x)

		x = self.norm(x)
		return x, atten_map

	def forward_part_features(self, x_parts, part_split):
		B, N, C = x_parts.shape
		parts_idx, parts_weight = part_split

		part_features = []
		for i in range(self.part_num):
			parts_feat = []
			for j in range(B):
				part = x_parts[j, parts_idx[i][j]]	# N, C
				part_feat = torch.sum(part * parts_weight[i][j].unsqueeze(1), dim=0)
				parts_feat.append(part_feat)
			part_features.append(torch.stack(parts_feat, dim=0))
		part_features = torch.stack(part_features, dim=2)	# B, C, P
		return part_features

	def forward_part_head(self, x_parts):
		part_out = []
		for i in range(self.part_num):
			part_head = getattr(self, f'part_head_{i}')
			output = part_head(x_parts[:,:,i])
			part_out.append(output)
		return part_out

	def forward_global_head(self, x):
		return self.global_head(x[:, 0])

	def forward(self, x, path=None):
		x, atten_map = self.forward_features_atten(x)

		if self.part_num > 0:
			part_split = self.get_part_split(atten_map)
			x_parts = self.forward_part_features(x[:, 1:], part_split)
			x_parts = self.forward_part_head(x_parts)
		else:
			x_parts = []

		x = self.forward_global_head(x)

		if self.training:
			list_class, list_feat = [], []
			for i in (x_parts + [x]):
				list_class.append(i[0])
				list_feat.append(i[1])
			return torch.stack(list_class, dim=2), torch.stack(list_feat, dim=2)	# B, C, P
		else:
			return torch.stack(([x] + x_parts), dim=2)	# B, C, P

	def get_part_split(self, att):
		B, H, _, _ = att.shape
		grid_size = self.patch_embed.grid_size
		N = grid_size[0] * grid_size[1]

		att_map = att[:, :, 0, 1:].mean(dim=1)
		att_map_argsort = torch.argsort(att_map, dim=1, descending=True)
		split_list = self.get_split_list(N, self.part_num)

		parts_idx = att_map_argsort.split(split_list, dim=1)
		parts_weight = []
		for part in parts_idx:
			part_weight = torch.zeros(part.shape).cuda()
			for i in range(B):
				part_weight[i] = att_map[i, part[i]]
			parts_weight.append(part_weight.softmax(dim=1))
		
		return parts_idx, parts_weight

	def get_split_list(self, total_num, group_num):
		split_list = torch.FloatTensor(range(group_num)).softmax(dim=0) * total_num
		split_list = split_list.round()
		if split_list.sum() != total_num:
			split_list[-1] += total_num - split_list.sum()
		return split_list.long().tolist()


def vit_small_patch16_224(args, img_size, pretrained=False, **kwargs):
	model_kwargs = dict(
		part_num = args.part_num,
		ft_dim = args.ft_dim,

		num_classes = args.num_classes,
		img_size = img_size,
		
		patch_size = 16,
		embed_dim = 384,
		depth = 12,
		num_heads = 6,
	)
	model = Vit(**model_kwargs, **kwargs)
	model.default_cfg = _cfg()
	
	if pretrained:
		url = 'https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'
		checkpoint_path = download_cached_file(url)
		model.load_pretrained(checkpoint_path)

	return model
