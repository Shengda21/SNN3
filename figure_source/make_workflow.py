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
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="790" viewBox="0 0 1200 790">',
 '<rect x="0" y="0" width="1200" height="790" fill="white"/>']
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

text(25,28,'(a) Matched adaptation opportunities',25,True,'start')
text(25,60,'37,028 training blocks  |  128 holdout blocks for learning-rate selection',20,False,'start')
node('fused_initialization',25,118,210,140,['SmoothSpike','Fused weights θ₀','H₁ fixed'])
node('T2_search',285,95,240,96,['T = 2 search','4 rates × 2 seeds','2,048 updates'], '#E9F4FA','#176B91')
node('T4_search',285,205,240,96,['T = 4 search','4 rates × 2 seeds','2,048 updates'], '#FBEFE5','#C46A20')
node('T2_adaptation',565,95,230,96,['T = 2 adaptation','3 fresh seeds','4,096 updates'], '#E9F4FA','#176B91')
node('T4_adaptation',565,205,230,96,['T = 4 adaptation','3 fresh seeds','4,096 updates'], '#FBEFE5','#C46A20')
node('model_lock',855,112,320,166,['Lock models + analyses','Frozen T = 2 and T = 4','Paired adapted weights','Time-cap checkpoints'], '#EDF3EE','#4F7460')
line([(235,170),(260,170),(260,136),(285,136)],'#176B91')
line([(235,207),(260,207),(260,253),(285,253)],'#C46A20')
line([(525,136),(565,136)],'#176B91');line([(525,253),(565,253)],'#C46A20')
line([(795,136),(855,136)],'#176B91');line([(795,253),(855,253)],'#C46A20')
line([(130,258),(130,338),(895,338),(895,278)],dashed=True)
text(455,329,'Retain the frozen comparator',19)
line([(1070,278),(1070,399)],'#4F7460')
text(1083,371,'After lock',19,False,'start','#4F7460')
line([(985,278),(985,358),(12,358),(12,518),(25,518)])

text(25,389,'(b) Execution comparisons for each checkpoint',25,True,'start')
node('resident_input_and_weights',25,441,214,153,['Same checkpoint','Fixed input shape','GPU input buffers','Neuron reset'], '#EDF3F6')
node('eager_execution',300,443,235,66,['Eager execution','Operator dispatch'])
node('graph_execution',300,568,235,66,['CUDA Graph replay','Captured operations'])
node('preserved_outputs',590,482,235,99,['Within-shape check','A / B / A inputs','Bitwise logit equality'], '#EDF3EE','#4F7460')
node('test_quality',865,399,310,67,['Test quality: eager, B = 4','4,406 blocks; fixed masks'], '#EDF3EE','#4F7460')
node('latency_measurement',865,522,310,92,['Median batch latency','B = 1, 4, 16, 64','30 calls per engine'])
line([(239,496),(269,496),(269,476),(300,476)])
line([(239,552),(269,552),(269,601),(300,601)])
line([(535,476),(560,476),(560,507),(590,507)])
line([(535,601),(560,601),(560,552),(590,552)])
line([(825,531),(845,531),(845,556),(865,556)])
line([(1175,432),(1190,432),(1190,656),(1098,656),(1098,698)],'#4F7460')
line([(1020,614),(1020,698)])
node('joint_decision',865,698,310,67,['Quality and latency','Fixed scoring protocol'], '#E9EDF5','#576687')
text(25,688,'Training contrast',21,True,'start')
text(25,719,'Adapted T2 vs frozen T4',20,False,'start')
text(25,748,'Adapted T2 vs adapted T4',20,False,'start')
text(439,688,'Execution contrast',21,True,'start')
text(439,719,'T2 vs T4: same engine',20,False,'start')
text(439,748,'T2 eager vs T4 graph',20,False,'start')
parts.append('</svg>');svg='\n'.join(parts)
(OUT/'workflow.svg').write_text(svg,encoding='utf-8')
print('SVG created; exporting vectors.',flush=True)
draw=svg2rlg(str(OUT/'workflow.svg'),font_map=font_map)
STATE_DEFAULTS['fontName']='DejaVu Sans'
c=canvas.Canvas(str(OUT/'workflow.pdf'),pagesize=(draw.width,draw.height),initialFontName='DejaVu Sans')
renderPDF.draw(draw,c,0,0);c.showPage();c.save()
doc=fitz.open(OUT/'workflow.pdf');doc[0].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(OUT/'workflow.png')
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
audit={'generation':'pure SVG generation','nodes':len(nodes),'connectors':len(edges),'all_connectors_orthogonal':True,
 'node_overlaps':overlaps,'text_overlaps':text_collisions,'connector_text_intersections':edge_text_collisions,
 'node_text_padding_failures':padding_failures,'minimum_font_size_svg_units':min(t['size'] for t in texts),
 'raster_images':svg.count('<image'),'live_text_elements':len(texts)}
(ROOT/'support/workflow_geometry.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
assert not (overlaps or text_collisions or edge_text_collisions or padding_failures),audit
print(audit)
faulthandler.cancel_dump_traceback_later()
