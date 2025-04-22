import torch

feature = torch.load("./data/things-eeg/Image_feature/FoveaBlur/intra-subject_ubp_EEGProjectLayer_RN50_test.pt", weights_only=False)
print(feature.keys())
print(feature['img_features'].keys())
print(feature['text_features'].keys())