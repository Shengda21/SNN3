"""Pure SVG generation with live text and orthogonal routed connectors."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import STATE_DEFAULTS
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from svglib.fonts import register_font, get_global_font_map
from matplotlib.font_manager import findfont, FontProperties
import fitz
import faulthandler
faulthandler.dump_traceback_later(20)

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures'
for bold in [False,True]:
 name='DejaVu Sans'+('-Bold' if bold else '')
 font_path=findfont(FontProperties(family='DejaVu Sans',weight='bold' if bold else 'normal'))
 pdfmetrics.registerFont(TTFont(name,font_path))
 register_font('DejaVu Sans',font_path=font_path,weight='bold' if bold else 'normal',rlgFontName=name)
pdfmetrics.registerFontFamily('DejaVu Sans',normal='DejaVu Sans',bold='DejaVu Sans-Bold',italic='DejaVu Sans',boldItalic='DejaVu Sans-Bold')
font_map=get_global_font_map()
font_map.find_font=lambda font_name,weight='normal',style='normal': ('DejaVu Sans-Bold' if str(weight).lower() in ['bold','700','800','900'] else 'DejaVu Sans',True)
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1020" viewBox="0 0 1200 1020">',
 '<rect x="0" y="0" width="1200" height="1020" fill="white"/>']
nodes=[];edges=[];texts=[]
def text(x,y,s,size=21,bold=False,anchor='middle',color='#23313C'):
 parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="DejaVu Sans" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{escape(s)}</text>')
 font='DejaVu Sans-Bold' if bold else 'DejaVu Sans'
 width=pdfmetrics.stringWidth(s,font,size)
 left=x-width/2 if anchor=='middle' else (x-width if anchor=='end' else x)
 ascent,descent=pdfmetrics.getAscentDescent(font,size)
 texts.append({'x':x,'baseline':y,'text':s,'size':size,'anchor':anchor,
               'bbox':[left,y-ascent,left+width,y-descent]})
def node(id,x,y,w,h,lines,color='#EDF3F6',stroke='#627789'):
 nodes.append({'id':id,'x':x,'y':y,'w':w,'h':h})
 parts.append(f'<g id="{id}"><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" stroke="{stroke}" stroke-width="1.8"/>')
 start=y+h/2-(len(lines)-1)*14+7
 size=min([21]+[(w-24)/pdfmetrics.stringWidth(l,'DejaVu Sans-Bold' if k==0 else 'DejaVu Sans',1) for k,l in enumerate(lines)])
 for k,l in enumerate(lines):
  text(x+w/2,start+k*28,l,size,k==0)
  texts[-1]['node']=id
 parts.append('</g>')
def line(points,c='#526371',arrow=True,dashed=False):
 assert all(x1==x2 or y1==y2 for (x1,y1),(x2,y2) in zip(points,points[1:])),points
 edges.append(points)
 ps=' '.join(f'{x},{y}' for x,y in points)
 dash=' stroke-dasharray="6 5"' if dashed else ''
 parts.append(f'<polyline points="{ps}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
 if arrow:
  (a,b),(x,y)=points[-2:]
  if x==a:
   d=1 if y>b else -1; p=[(x,y),(x-5,y-9*d),(x+5,y-9*d)]
  else:
   d=1 if x>a else -1;p=[(x,y),(x-9*d,y-5),(x-9*d,y+5)]
  parts.append(f'<polygon points="{" ".join(f"{a},{b}" for a,b in p)}" fill="{c}"/>')

text(25,28,'(a) Shared initialization and paired adaptation',25,True,'start')
node('initial_state',25,81,210,138,['Shared state S₀','Four-step asset','Parameters','and buffers'])
node('adapt_T2',295,65,245,66,['Adapt at T = 2','Fixed update budget'], '#E9F4FA','#176B91')
node('adapt_T4',295,169,245,66,['Adapt at T = 4','Fixed update budget'], '#FBEFE5','#C46A20')
node('state_T2',600,65,235,66,['State S₂(K, s)','Paired data stream'], '#E9F4FA','#176B91')
node('state_T4',600,169,235,66,['State S₄(K, s)','Paired data stream'], '#FBEFE5','#C46A20')
node('model_lock',900,81,275,138,['Lock states + analyses','Both budgets','All scoring paths'])
line([(235,117),(265,117),(265,98),(295,98)],'#176B91')
line([(235,182),(265,182),(265,202),(295,202)],'#C46A20')
line([(540,98),(600,98)],'#176B91');line([(540,202),(600,202)],'#C46A20')
line([(835,98),(900,98)],'#176B91');line([(835,202),(900,202)],'#C46A20')
line([(130,219),(130,273),(1037,273),(1037,219)],dashed=True)
text(580,265,'Retain S₀ at both inference windows',20)
text(25,303,'Language: SmoothSpike / WikiText-2',21,True,'start')
text(585,303,'3 paired seeds; K = 512 and 4,096',21,False,'start')
text(25,337,'Vision: SpikingResformer / CIFAR-100',21,True,'start')
text(585,337,'5 paired seeds; K = 512 and 2,048',21,False,'start')

text(25,389,'(b) Cross-evaluation at each budget',25,True,'start')
parts.append('<rect x="25" y="407" width="710" height="267" fill="#FAFBFC" stroke="#AAB5BC" stroke-width="1.3"/>')
text(302,432,'Inference T = 2',21,True)
text(561,432,'Inference T = 4',21,True)
text(95,484,'S₀',23,True)
text(95,561,'S₂',23,True,color='#176B91')
text(95,638,'S₄',23,True,color='#C46A20')
node('q02',190,450,224,53,['Q₀(2)'])
node('q04',449,450,224,53,['Q₀(4)'])
node('q22',190,527,224,53,['Q(2, 2)'], '#E9F4FA','#176B91')
node('q24',449,527,224,53,['Q(2, 4)'], '#E9F4FA','#176B91')
node('q42',190,604,224,53,['Q(4, 2)'], '#FBEFE5','#C46A20')
node('q44',449,604,224,53,['Q(4, 4)'], '#FBEFE5','#C46A20')
text(340,699,'P₂ = Q(4, 2) − Q(2, 2)',21)
text(340,730,'P₄ = Q(2, 4) − Q(4, 4)',21)
node('fixed_test',805,420,370,77,['Fixed test inputs; batch = 4','Same samples for every state'])
line([(1175,180),(1190,180),(1190,401),(990,401),(990,420)])
line([(805,459),(762,459),(762,476),(735,476)])
node('raw_state_path',805,530,370,60,['Raw states','Language + vision'])
line([(805,560),(735,560)])
node('bn_calibration',805,649,370,93,['Vision BN-only states','2,048 training images','Only BN statistics change'], '#EDF3EE','#4F7460')
line([(990,590),(990,649)],'#4F7460')
text(1004,625,'Vision only',20,False,'start','#4F7460')
line([(805,700),(762,700),(762,631),(735,631)],'#4F7460')

text(25,799,'(c) Quality and latency for final native states',25,True,'start')
node('final_native',25,858,225,94,['Final S₂ at T = 2','Final S₄ at T = 4','Paired seed / GPU'])
node('eager_route',315,837,230,63,['Eager forward','GPU-resident input'])
node('graph_route',315,932,230,63,['CUDA Graph','GPU-resident input'])
node('full_measurements',620,853,255,105,['Full test scoring','3 processes per pair','30 calls per engine'])
node('quality_latency',940,862,235,90,['Measured loss','Median latency','Process variation'])
line([(250,877),(280,877),(280,869),(315,869)])
line([(250,934),(280,934),(280,964),(315,964)])
line([(545,869),(580,869),(580,877),(620,877)])
line([(545,964),(580,964),(580,934),(620,934)])
line([(875,906),(940,906)])
parts.append('</svg>');svg='\n'.join(parts)
(OUT/'workflow_revision2.svg').write_text(svg,encoding='utf-8')
print('SVG created; exporting vectors.',flush=True)
draw=svg2rlg(str(OUT/'workflow_revision2.svg'),font_map=font_map)
STATE_DEFAULTS['fontName']='DejaVu Sans'
c=canvas.Canvas(str(OUT/'workflow_revision2.pdf'),pagesize=(draw.width,draw.height),initialFontName='DejaVu Sans')
renderPDF.draw(draw,c,0,0);c.showPage();c.save()
doc=fitz.open(OUT/'workflow_revision2.pdf');doc[0].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(OUT/'workflow_revision2.png')
overlaps=[]
for i,a in enumerate(nodes):
 for b in nodes[i+1:]:
  if max(a['x'],b['x'])<min(a['x']+a['w'],b['x']+b['w']) and max(a['y'],b['y'])<min(a['y']+a['h'],b['y']+b['h']):overlaps.append([a['id'],b['id']])
def intersects(a,b):
 return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])
text_collisions=[];edge_text_collisions=[];padding_failures=[]
for i,a in enumerate(texts):
 for b in texts[i+1:]:
  if intersects(a['bbox'],b['bbox']):text_collisions.append([a['text'],b['text']])
 if 'node' in a:
  n=next(n for n in nodes if n['id']==a['node']);left,top,right,bottom=a['bbox']
  if min(left-n['x'],n['x']+n['w']-right)<11.9 or min(top-n['y'],n['y']+n['h']-bottom)<3:
   padding_failures.append(a['text'])
 for j,e in enumerate(edges):
  for (x1,y1),(x2,y2) in zip(e,e[1:]):
   segment=[min(x1,x2)-1,min(y1,y2)-1,max(x1,x2)+1,max(y1,y2)+1]
   if intersects(a['bbox'],segment):edge_text_collisions.append([j,a['text']])
out_of_bounds=[a['text'] for a in texts if a['bbox'][0]<0 or a['bbox'][1]<0 or a['bbox'][2]>1200 or a['bbox'][3]>1020]
audit={'generation':'pure SVG generation','nodes':len(nodes),'connectors':len(edges),'all_connectors_orthogonal':True,
 'node_overlaps':overlaps,'text_overlaps':text_collisions,'connector_text_intersections':edge_text_collisions,
 'node_text_padding_failures':padding_failures,'minimum_font_size_svg_units':min(t['size'] for t in texts),
 'raster_images':svg.count('<image'),'live_text_elements':len(texts),'out_of_bounds_text':out_of_bounds}
(ROOT/'support/workflow_revision2_geometry.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
assert not (overlaps or text_collisions or edge_text_collisions or padding_failures or out_of_bounds),audit
print(audit)
faulthandler.cancel_dump_traceback_later()
