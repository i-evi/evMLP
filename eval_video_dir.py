import os
import cv2
import sys
import torch
import evmlp
import event_driven
from tqdm import tqdm
from torchvision import transforms
from thop import profile

if len(sys.argv) != 4:
    print("Usage: python eval_video_dir.py <weights.pth> <dir_path> <event_threshold>")
    sys.exit(1)

weights_file = sys.argv[1]
video_path = sys.argv[2]
event_threshold = float(sys.argv[3])

_use_cuda = False
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    _use_cuda = True

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def filter_video_files(file_list):
    video_extensions = {'.avi', '.mp4'}
    return [file for file in file_list if any(file.endswith(ext) for ext in video_extensions)]

file_list = filter_video_files(os.listdir(video_path))

net = evmlp.evMLP()
if _use_cuda:
    net.load_state_dict(torch.load(weights_file))
    net.cuda()
else:
    net.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))

net.eval()

event_driven_evmlp = event_driven.EventDrivenEvMLP(net, event_threshold, device=DEVICE)
init_image = torch.zeros([224, 224, 3], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)
result, macs = event_driven_evmlp.eval(init_image)

macs_without_event = macs
macs_accumulator = 0.
correct_pred = 0
frame_counter = 0

with torch.no_grad():

    for item in tqdm(file_list, desc="Progress"):
        file_pathname = ("%s/%s"%(video_path, item))
        cap = cv2.VideoCapture(file_pathname)
        print("open:", file_pathname, end="")

        for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1)):
            ret, frame = cap.read()
            if not ret:
                print("finished: ", file_pathname)
                avg_macs = macs_accumulator / frame_counter
                cap.release()
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tsr = transform(frame).unsqueeze(0)
            frame_tsr = frame_tsr.to(DEVICE)
     
            gt = net(frame_tsr)
            result, macs = event_driven_evmlp.eval(frame_tsr)
            macs_accumulator = macs_accumulator + macs
            frame_counter = frame_counter + 1
            # top5_gt     = torch.topk(gt, 5)
            # top5_result = torch.topk(result, 5)
            if gt.topk(1).indices == result.topk(1).indices:
                correct_pred = correct_pred + 1

        avg_macs = macs_accumulator / frame_counter
        print("\nTotal frames processed:%d,"%frame_counter, "avg MACs: %f,"%avg_macs,
            "saving MACs by %f%%,"%(((macs_without_event - avg_macs) / macs_without_event) * 100),
            "accuracy top1: %f"%(correct_pred / frame_counter))

print("macs_without_event", macs_without_event,
    "macs_accumulator:", macs_accumulator, "correct_pred:",
    correct_pred, "frame_counter:", frame_counter)

