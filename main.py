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
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filepath = Path(args.filepath) if args.filepath is not None else None
    ckpt_path = Path(args.checkpoint)
    use_camera = args.use_camera
    print(filepath if not use_camera else "CAMERA", use_camera, args.use_cpu)

    model = Classifire(arch="resnet18", pretrain=False, num_classes=2)
    if torch.cuda.is_available() and not args.use_cpu:
        device = "cuda"
        model = model.cuda()
    else:
        device = "cpu"
    print(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = preprocess_transform()

    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        assert filepath is not None, "Please input filepath."
        assert filepath.exists(), f"filepath: {str(filepath)} does not exists."
        cap = cv2.VideoCapture(str(filepath))
        movie_fps = int(cap.get(cv2.CAP_PROP_FPS))
        movie_fps_time = 1 / (movie_fps * 1)

    loop_tm = cv2.TickMeter()
    tm = cv2.TickMeter()
    tm.start()
    max_count = 15
    count = 0
    fps = 0

    with torch.no_grad():
        while True:
            loop_tm.stop()
            loop_tm.reset()
            loop_tm.start()
            if count == max_count:
                tm.stop()
                fps = max_count / tm.getTimeSec()
                tm.reset()
                tm.start()
                count = 0
            ret, frame = cap.read()
            h, w, c = frame.shape
            # result = DNN(frame)
            tensor = transform(frame).to(device)
            output = model(tensor.unsqueeze(0))
            pred = torch.argmax(output, dim=1).detach().cpu()
            print(output.detach().cpu().tolist(), pred.item(), fps)


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

            cv2.putText(frame, 'FPS: {:.2f}'.format(fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

            cv2.imshow("Show FLAME Image", frame)
            count += 1
            if not use_camera:
                pass
                #time.sleep(movie_fps_time - loop_tm.getTimeSec())

            k = cv2.waitKey(10)
            if k == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
