# PraNet upstream source notice

Repository: [DengPingFan/PraNet](https://github.com/DengPingFan/PraNet).
Pinned commit: `95f12bd4e32ec954b5069654abae7de9473fd1f5`.

The [pinned upstream README, section 7](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/README.md#7-license)
states:

> The source code is free for research and education use only. Any comercial use should get formal permission first.

The spelling above is retained from the original. This statement is recorded
for the PraNet/Res2Net sources supplied by that repository; it is not replaced
with an MIT, BSD or Apache license for the whole project.

The upstream files `lib/PraNet_Res2Net.py`, `lib/Res2Net_v1b.py` and
`MyTrain.py` are obtained separately by the user. Their model/loss definitions
are loaded through `src/dirags/integrations/backbone_1.py`; their implementations
are not included in this methods-only edition. Parameter names and model/loss
calculations remain upstream definitions.

The formal factory in `cross_backbone/backbone_1.py` supplies the pretrained state
at runtime. It bypasses the upstream constructor's hard-coded local pretrained
path, as the original experiment adapter did. Direct execution of upstream
training scripts is not the study's reproduction entrypoint. Cite PraNet and retain the
upstream acknowledgements and citation information linked above when using
this backbone.
