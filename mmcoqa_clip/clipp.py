# import clip
# import torch
# import numpy as np
# from PIL import Image
# import sys
# sys.path.append('../mae')
# import models_mae
# def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
#     # build model
#     model = getattr(models_mae, arch)()
#     # load model
#     checkpoint = torch.load(chkpt_dir, map_location='cpu')
#     msg = model.load_state_dict(checkpoint['model'], strict=False)
#     print(msg)
#     return model
# chkpt_dir = '../../mae_visualize_vit_large.pth'
# model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16').to("cuda:7")
# layer = torch.nn.Linear(1024, 512).to("cuda:7")
# # model, preprocess = clip.load("ViT-B/32", device="cuda:7")
# img = torch.rand(1, 3, 224, 224).to("cuda:7")
# with torch.no_grad():
#     res = layer(model_mae.forward_encoder(img, 0)[0])
#     print(res.shape)
#     # # print(model.input_resolution)
# # model = model.float()
# # img = preprocess(Image.fromarray(img)).unsqueeze(0).to("cuda:1")
# # print(img.shape)
# # print(model.visual(img).shape)