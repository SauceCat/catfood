# CenterNet-104

Almost the same as CornerNet, only an extra center keypoint.

```python
class model(kp):
	# make_kp_layer
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(k=3, inp_dim=256, out_dim=256, stride=1, with_bn=False),
            nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=(1, 1)),
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=(1, 1), bias=False
            ),
            nn.BatchNorm2d(256),
        )

    def __init__(self):
        stacks = 2
        pre = nn.Sequential(
            convolution(k=7, inp_dim=3, out_dim=128, stride=2, with_bn=True),
            residual(inp_dim=128, out_dim=256, k=3, stride=2),
        )
        hg_mods = nn.ModuleList(
            [
                hg_module(
                    n=5,
                    dims=[256, 256, 384, 384, 384, 512],
                    modules=[2, 2, 2, 2, 2, 4],
                    make_pool_layer=make_pool_layer,
                    make_hg_layer=make_hg_layer,
                )
                for _ in range(stacks)
            ]
        )

        cnvs = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        cnvs_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
		
        # hourglass
        hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_)
		
        # CornerNet diff
        # corner pooling
        tl_modules = nn.ModuleList(
            [pool(256, TopPool, LeftPool) for _ in range(stacks)]
        )
        br_modules = nn.ModuleList(
            [pool(256, BottomPool, RightPool) for _ in range(stacks)]
        )
        # center pooling
        ct_modules = nn.ModuleList([
            pool_cross(
                256, TopPool, LeftPool, BottomPool, RightPool
            ) for _ in range(stacks)
        ])
		
        # CornerNet diff
        # heatmaps
        tl_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        br_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        ct_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        for tl_heat, br_heat, ct_heat in zip(tl_heats, br_heats, ct_heats):
            torch.nn.init.constant_(tl_heat[-1].bias, -2.19)
            torch.nn.init.constant_(br_heat[-1].bias, -2.19)
            torch.nn.init.constant_(ct_heat[-1].bias, -2.19)
		
        # embeddings
        tl_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        br_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
		
        # offsets
        tl_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
        br_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
        ct_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
		
        # hg_net
        super(model, self).__init__(
            hgs,
            tl_modules,
            br_modules,
            ct_modules,
            tl_heats,
            br_heats,
            ct_heats,
            tl_tags,
            br_tags,
            tl_offs,
            br_offs,
            ct_offs
        )
		
        # loss
        self.loss = AELoss(pull_weight=1e-1, push_weight=1e-1, focal_loss=_neg_loss)
```
