from flask import Flask,render_template,Response
import cv2
import subprocess

app = Flask(__name__)

video_path = 'e:\\coding\\graduation-project\\find-the-fight\\server\\violence-10.mp4'
inference_video_path = 'e:\\coding\\graduation-project\\AlphaPose\\examples\\res\\AlphaPose_violence-10.mp4'

def run_alpha_pose_script():
  script_path = 'e:/coding/graduation-project/AlphaPose/scripts/demo_inference.py'
  arg1 = 'scripts/demo_inference.py'
  arg2 = '--format'
  arg3 = 'open'
  arg4 = '--cfg'
  arg5 = 'configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml'
  arg6 = '--checkpoint'
  arg7 = 'pretrained_models/fast_res50_256x192.pth'
  arg8 = '--video'
  arg9 = 'examples/demo/fight/violence-10.mp4'
  arg10 = '--outdir'
  arg11 = 'examples/res/fight'
  arg12 = '--save_img'

  subprocess.run(['CALL conda.bat', 'activate', 'alphapose'])
  subprocess.run(['C:/Users/mateu/miniconda3/envs/alphapose/python.exe', script_path, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12])
  return

def generate_frames():
  # run_alpha_pose_script()
  inference_video_path = 'e:\\coding\\graduation-project\\AlphaPose\\examples\\res\\AlphaPose_violence-10.mp4'
  camera = cv2.VideoCapture(inference_video_path)
  while True:
    ## read the camera frame
    success, frame = camera.read()
    if not success:
      break
    else:
      ret, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()

    yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
