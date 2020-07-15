""" Detect people wearing masks in videos
"""
from pathlib import Path

import click
import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from common.facedetector import FaceDetector
from train import MaskDetector


@click.command(help="""
                    modelPath: path to model.ckpt\n
                    videoPath: path to video file to annotate
                    """)
@click.argument('modelpath')
@click.argument('videopath')
@click.option('--output', 'outputPath', type=Path,
              help='specify output path to save video with annotations')
@torch.no_grad()
def tagVideo(modelpath, videopath, outputPath=None):
    """ detect if persons in video are wearing masks or not
    """
    result=-1
    model = MaskDetector()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(modelpath,map_location=device)['state_dict'], strict=False)
    
    model = model.to(device)
    model.eval()
    
    faceDetector = FaceDetector(
        prototype='models/deploy.prototxt.txt',
        model='models/res10_300x300_ssd_iter_140000.caffemodel',
    )
    
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])
    
    if outputPath:
        writer = FFmpegWriter(str(outputPath))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]
    for frame in vreader(str(videopath)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceDetector.detect(frame)
        print ("FRAME")
        for face in faces:
            xStart, yStart, width, height = face
            print ("FACE",face)
            
            # clamp coordinates that are outside of the image
            xStart, yStart = max(xStart, 0), max(yStart, 0)

            # Image is 640x640
            print ("DIMS",xStart, yStart ,width,height)
            right = min(xStart+width,639)
            bottom = min(yStart+height,639)
            print ("Right",xStart+width)
            print ("Bottom",yStart+height)
           
            area = width*height
            
            inarea = (right-xStart)*(bottom-yStart)
            print ("Area",area)
            print ("inArea",inarea)
            areaperc=inarea/area
            #print ("areaperc",areaperc)
            
            # predict mask label on extracted face
            faceImg = frame[yStart:yStart+height, xStart:xStart+width]
            output = model(transformations(faceImg).unsqueeze(0).to(device))
            print ("OUTPUT",output)
            _, predicted = torch.max(output.data, 1)
            print ("result",_)
            
            # draw face frame
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=2)
            
            # center text according to the face frame
            textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0] // 2
            
            # draw prediction label
            if (areaperc > 0.75):
              print (labels[predicted])
              if predicted:
                result=1
              elif result == -1:
                result=0
            else:
              print ("Insufficuent")
            cv2.putText(frame,
                        labels[predicted],
                        (textX, yStart-20),
                        font, 1, labelColor[predicted], 2)
        if outputPath:
            writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        #cv2.imshow('main', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    if outputPath:
        writer.close()
    #cv2.destroyAllWindows()
    if result==0: print ("No Mask Found")
    if result==1: print ("Mask Found")
    if result==-1: print ("No face in photo")
    return result

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    tagVideo()
