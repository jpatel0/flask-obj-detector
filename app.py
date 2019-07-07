from flask import Flask, render_template, request,redirect,url_for,send_file, Response
import cv2
from camera import VideoCamera
from darkflow.net.build import TFNet
from darkflow.defaults import argHandler
import os
from PIL import Image



if("uploaded" not in os.listdir("./static/")):
    os.mkdir("static/uploaded")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join("static","uploaded")
app.config['SECRET_KEY'] = 'kjabsbgbfsagbfsb#'

def initModel():
    global tfnet
    lst = ["flow","--model", "cfg/tiny-yolo-voc.cfg", "--load", "bin/tiny-yolo-voc.weights", "--threshold", "0.47", '--gpu', '0.7']
        # lst = ['--model', 'cfg/tiny-yolo-voc-1c.cfg', '--load', '125', '--demo', 'camera', '--gpu', '0.7']

    flags = argHandler()
    flags.setDefaults()
    flags.parseArgs(lst)
    flags.demo = None
    try: 
        flags.load = int(flags.load)
    except: 
        pass

    tfnet = TFNet(flags)


@app.route("/")
def index():
    return render_template("index.html")


def gen(camera):
    global tfnet
    while True:
        frame1 = camera.get_frame()
    
        pre = tfnet.framework.preprocess(frame1)
        buff_pre = [pre]
        out = tfnet.sess.run(tfnet.out,{tfnet.inp:buff_pre})

        postprocessed = tfnet.framework.postprocess(out[0], frame1, False)
        # canera.release()

        _, jpeg = cv2.imencode('.jpg', postprocessed)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test')
def test():
    return render_template('test.html')


@app.route("/upload",methods=["POST","GET"])
def upload():
    if(request.method == 'POST' and request.form.get('vid')!=None):
        return render_template("uploadVid.html")
    
    elif(request.method == 'POST' and request.form.get('cam')!=None):
        return render_template("uploadImg.html",cap="capture")

    return render_template("upload.html",cap="")

@app.route("/upload_img",methods=["POST"])
def upload_img():
    global tfnet
    accept_ext = ['jpg','jpeg','png']
    if('file' in request.files and any(True for ext in accept_ext if ext in request.files['file'].filename)):
        # return "File accepted"
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(filepath)
        imgcv = cv2.imread(filepath)
        pre = tfnet.framework.preprocess(imgcv)
        buff_pre = [pre]
        out = tfnet.sess.run(tfnet.out,{tfnet.inp:buff_pre})
        postprocessed = tfnet.framework.postprocess(out[0], imgcv, False)
        cv2.imwrite(filepath,postprocessed)
        return render_template("pred.html",filepath=filepath)
    else:
        return "Uploaded file is not accepted"

@app.route("/upload_vid",methods=["POST"])
def upload_vid():
    global tfnet
    if('vidfile' in request.files):
        # return "File accepted"
        file = request.files['vidfile']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file_name_ext = os.path.splitext(file.filename)
        newfilename = file_name_ext[0]+"_pred.avi"
        filechangepath = os.path.join(app.config['UPLOAD_FOLDER'],newfilename)
        print(filechangepath)
        file.save(filepath)
        vid = cv2.VideoCapture(filepath)
        assert vid.isOpened(), \
            'Cannot capture source'
        _, frame = vid.read()
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = round(vid.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(filechangepath, fourcc, fps, (width, height))
        print("[INFO] Processing the video file...")
        while vid.isOpened():
            ret, imgcv = vid.read()
            if(not ret):
                break
            pre = tfnet.framework.preprocess(imgcv)
            buff_pre = [pre]
            out = tfnet.sess.run(tfnet.out,{tfnet.inp:buff_pre})

            postprocessed = tfnet.framework.postprocess(out[0], imgcv, False)
            videoWriter.write(postprocessed)
        print("[INFO] Saving the video file")
        videoWriter.release()
        vid.release()
        return send_file(filechangepath,attachment_filename=newfilename,as_attachment=True)
    else:
        return "Uploaded file is not accepted"


if __name__ == "__main__":
    initModel()
    
    app.run(host="0.0.0.0")
    