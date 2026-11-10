"""Compose actual native free RGB/Alpha, using the fixed shared camera protocol."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'comparison/visualizations/scientific_review/free';DEST.mkdir(parents=True,exist_ok=True)
protocol=ROOT/'comparison/shared_free_views.json';views=json.loads(protocol.read_text())['views']
records=[]
for view in views:
 name=view['name'];qn=name.replace('+','p').replace('-','m')
 recipes=[]
 for label,folder in [('E10',ROOT/'comparison/visualizations/E10_free'),('P2_Cleaned',ROOT/'comparison/visualizations/P2_free'),('E11_native2DGS',ROOT/'comparison/visualizations/E11_free'),('E12_RAW_no_EO',ROOT/'comparison/visualizations/E12_free')]:
  recipes.append((label,folder/name/'RGB.png',folder/name/'Alpha.png'))
 recipes.append(('E13_QGS',ROOT/'E13_QGS/free_views'/(qn+'_rgb_gray_background.png'),ROOT/'E13_QGS/free_views'/(qn+'_alpha.png')))
 recipes.append(('E14_2D_30K',ROOT/'E14_2D_SuGaR/visual_views/surfel/final_30000'/name/'rgb_gray.png',ROOT/'E14_2D_SuGaR/visual_views/surfel/final_30000'/name/'alpha.png'))
 recipes.append(('E14_Refined',ROOT/'E14_2D_SuGaR/visual_views/refined/refined'/name/'rgb_gray.png',ROOT/'E14_2D_SuGaR/visual_views/refined/refined'/name/'alpha.png'))
 selected=[r for r in recipes if r[1].is_file() and r[2].is_file()]
 if not selected:continue
 panel=Image.new('RGB',(270*len(selected),1016),'#202020');draw=ImageDraw.Draw(panel);sources=[]
 for col,(label,rgb,alpha) in enumerate(selected):
  for row,(channel,p) in enumerate([('RGB_gray_background',rgb),('Alpha',alpha)]):
   panel.paste(Image.open(p).convert('RGB').resize((270,480),Image.Resampling.LANCZOS),(270*col,508*row+28))
   draw.text((270*col+3,508*row+5),label+' / '+channel,fill='white')
   sources.append(dict(method=label,channel=channel,path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 out=DEST/(name.replace('+','p').replace('-','m')+'_RGB_Alpha.png');panel.save(out)
 records.append(dict(view=name,output=str(out.resolve()),sources=sources,camera=view))
(DEST/'PROTOCOL.json').write_text(json.dumps(dict(status='actual_available_native_images',records=records,source_protocol=str(protocol.resolve()),source_protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),background=.25,EO='E12 uses actual raw Gaussian appearance; per-source EO does not define unseen-camera exposure',scope='Native renders only, no generated synthetic illustration. No novel-view photo ground truth; compare visible artifacts and Alpha, not novel-view PSNR.'),ensure_ascii=False,indent=2)+'\n')
print('Actual native free RGB/Alpha panels',len(records))
