"""Show native Original/Cleaned evidence and measured damage without a decision plug."""
import argparse,io,json
from PIL import Image,ImageDraw
from runtime import OUT,Meter,write


def main():
    p=argparse.ArgumentParser();p.add_argument('method');a=p.parse_args()
    folder=OUT/'cleaning'/a.method;meter=Meter();views=['0000','0048','0096','0144','0192','0208']
    before=json.loads((OUT/'gaussian'/a.method/'evaluation/metrics.json').read_text())['metrics']
    after=json.loads((folder/'evaluation/metrics.json').read_text())['metrics']
    write(folder/'comparison.json',dict(method=a.method,original=before,cleaned=after,delta_cleaned_minus_original={k:after[k]-before[k] for k in before if k in after},decision='Requires actual same-camera image review; no automatic acceptance'))
    for channel in ['RGB','Alpha','Depth','Normal_derived']:
        canvas=Image.new('RGB',(150+212*6,405*3),'#edf0f2');draw=ImageDraw.Draw(canvas)
        for row,label in enumerate(['Original RGB','Original Gaussian','C1_raw Cleaned']):
            y=row*405;draw.text((5,y+10),label,fill='black')
            for col,view in enumerate(views):
                path=OUT/'common_input/dataset/images'/f'{view}.jpg' if row==0 else (OUT/'gaussian'/a.method/'evaluation' if row==1 else folder/'evaluation')/f'{view}_{channel}.png'
                im=Image.open(path).convert('RGB');im.thumbnail((212,379));canvas.paste(im,(150+212*col,y+26));draw.text((154+212*col,y+5),view,fill='black')
        b=io.BytesIO();canvas.save(b,format='PNG');meter.save(folder/f'comparison_{channel}.png',b.getbuffer())


if __name__=='__main__':main()
