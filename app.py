from flask import Flask, render_template,url_for,redirect,flash,request
import torch
from torchvision import transforms
import io
from facenet_pytorch import MTCNN, extract_face
import os
from PIL import Image, ImageDraw, ImageColor
from werkzeug import secure_filename
import mmcv, cv2
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
workers = 0 if os.name == 'nt' else 4
frequency = 25
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','mp4','avi','mov'}

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300
# Specify a path
PATH = APP_ROOT + "/models/face_classification_v2.pt"

# Load
model = torch.load(PATH)
model.to(device)
model.eval()
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device,keep_all=True
)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if request.files['file'].filename == '':
            #flash('No selected file')
            return render_template('notselected.html',message = 'File is not selected')
            
        
        
        if 'jpg' in str(file.filename).lower() or 'jpeg' in str(file.filename).lower() or 'png' in str(file.filename).lower():
        
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)        
                filedir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filedir)
                filek = '/static/uploads/'+filename
                img = Image.open(filedir)
                boxes, probs = mtcnn.detect(img)
                # Draw boxes and save faces
                img_draw = img.copy()
                draw = ImageDraw.Draw(img_draw)
                if boxes is not None:
                    for j, box in enumerate(boxes):
                        extract_face(img, box, save_path='detected/detected_face_{}.png'.format(j)) 
                        with open(str(APP_ROOT) +'/detected/detected_face_{}.png'.format(j), 'rb') as f:
                            image_bytes = f.read()
                            pred_idx = get_prediction(image_bytes=image_bytes)
                            if int(pred_idx) == 0:
                                draw.rectangle(box.tolist(), width=10,outline=ImageColor.getrgb('green'))
                            else:
                                draw.rectangle(box.tolist(), width=10,outline=ImageColor.getrgb('red'))
                    img_draw.save(filedir)

                
        if 'mp4' in str(file.filename).lower() or 'mov' in str(file.filename).lower():
        
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filedir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filedir)
                filek = '/static/uploads/'+filename
                
                video = mmcv.VideoReader(filedir)
                frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
                last_predict = []
                box_list = []
                frames_tracked = []
                # loop through frames 
                for i, frame in enumerate(frames):
                    
                    width, height = frame.size
                    # Frequency to detect
                    if i % frequency == 0:
                        last_predict.clear()
                        # Detect faces
                        boxes, prob = mtcnn.detect(frame)
                        
                        box_list.append(boxes)
                        # Draw boxes and save faces
                        frame_draw = frame.copy()
                        draw = ImageDraw.Draw(frame_draw)
                        # Check if there is detection
                        if boxes is not None:
                            #Loop through all detections
                            for j, box in enumerate(boxes):
                                extract_face(frame, box, save_path='detected/detected_face_{}.png'.format(j))
                                with open(str(APP_ROOT) +'/detected/detected_face_{}.png'.format(j), 'rb') as f:
                                    image_bytes = f.read()
                                    pred_idx = get_prediction(image_bytes=image_bytes)
                                    last_predict.append(pred_idx)
                                    #blist = box.tolist()
                                    
                                    if int(pred_idx) == 0:
                                        draw.rectangle(box.tolist(), width=10,outline=ImageColor.getrgb('green'))
                                        #draw.text((blist[2], blist[3]), 'Mask', fill='green', font = ImageFont.truetype("/content/ArialMT.ttf",20))
                                        #draw.text((blist[2], blist[3]-30), str(prob), fill='green', font = ImageFont.truetype("/content/ArialMT.ttf",20))
                                    else:
                                        draw.rectangle(box.tolist(), width=10,outline=ImageColor.getrgb('red'))
                                        #draw.text((blist[2], blist[3]), 'Without mask', fill='red', font = ImageFont.truetype("/content/ArialMT.ttf",20))
                                        #draw.text((blist[2], blist[3]-30), str(prob), fill='red', font = ImageFont.truetype("/content/ArialMT.ttf",20))
                                       
                            frames_tracked.append(frame_draw.resize((width, height), Image.BILINEAR))
                        # If not detected            
                        else:
                            frames_tracked.append(frame_draw.resize((width, height), Image.BILINEAR))
                            
                    # If this is not the frequency to detect
                    else:
                        # If there were detections in previous frame
                        if box_list[-1] is not None:
                            boxes = box_list[-1]
                            # Draw boxes and save faces
                            frame_draw = frame.copy()
                            draw = ImageDraw.Draw(frame_draw)
                            
                            # If there were detections in previous frame
                            if boxes is not None:
                                
                                for j, box in enumerate(boxes):
                                    
                                    if int(last_predict[j]) == 0:
                                        draw.rectangle(box.tolist(), width=10,outline=ImageColor.getrgb('green'))
                                    else:
                                        draw.rectangle(box.tolist(), width=10,outline=ImageColor.getrgb('red'))
                                        
                                    
                        
                                frames_tracked.append(frame_draw.resize((width, height), Image.BILINEAR))
                        else:      
                                
                            # if no detections in previous frame add just a frame
                            frames_tracked.append(frame_draw.resize((width, height), Image.BILINEAR))
                                
                
                dim = frames_tracked[0].size
                fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
                video_tracked = cv2.VideoWriter(filename, fourcc, 25.0, dim)
                for frame in frames_tracked:
                    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                video_tracked.release()
                
                os.replace((str(APP_ROOT)+ '/' + filename), (str(APP_ROOT) +filek))
                
                                        
        return render_template('out.html', filek=filek)
            
    return render_template('index.html')
    
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(180),
                                        transforms.CenterCrop(160),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.6182, 0.5178, 0.4769],[0.2853, 0.2667, 0.2657])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port="5000")
