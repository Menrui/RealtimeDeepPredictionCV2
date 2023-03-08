import argparse
import time
from pathlib import Path

import cv2
import torch

from src.data_process import preprocess_transform
from src.resnet import Classifire


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("RECD")
    parser.add_argument(
        "--filepath", type=str, default="movie/WIN_20221122_15_59_43_Pro.mp4"
    )
    parser.add_argument("--checkpoint", type=str, default="model/best.pth")
    parser.add_argument("--use_camera", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filepath = Path(args.filepath) if args.filepath is not None else None
    ckpt_path = Path(args.checkpoint)
    use_camera = args.use_camera
    print(filepath, use_camera)

    model = Classifire(arch="resnet18", pretrain=False, num_classes=2)
    checkpoint = torch.load(str(ckpt_path))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
        model = model.cuda()
    else:
        device = "cpu"
    transform = preprocess_transform()

    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        assert filepath is not None, "Please input filepath."
        assert filepath.exists(), f"filepath: {str(filepath)} does not exists."
        cap = cv2.VideoCapture(str(filepath))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        FPS = 1 / (fps * 1)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            h, w, c = frame.shape
            # result = DNN(frame)
            tensor = transform(frame).to(device)
            output = model(tensor.unsqueeze(0))
            pred = torch.argmax(output, dim=1).detach().cpu()
            print(output, pred)

            if pred.item() == 1:
                cv2.putText(
                    frame,
                    text="CONTACT",
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    org=(100, 300),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    text="UNCONTACT",
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(255, 0, 255),
                    org=(100, 300),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            cv2.imshow("Show FLAME Image", frame)
            if not use_camera:
                time.sleep(FPS)

            k = cv2.waitKey(10)
            if k == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
