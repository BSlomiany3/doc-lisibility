from pipeline.functions import pipeline, VGGLikeNN
import torch

model = VGGLikeNN(63)
model.load_state_dict(torch.load("research/models/cnn_model2.pth"))

in_folder = "data/docs_samples/imgs"
test_file = "doc6.png"
out_folder = "pipeline/scored_outputs"


pipeline(in_folder, test_file, model, out_folder)