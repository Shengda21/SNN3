"""Update the existing vector experiment overview to the completed R3 design.

Deterministic SVG and TikZ from one geometry; black/white and straight arrows.
The manuscript is deliberately not modified by this generator.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import math
import subprocess
import fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib.font_manager import findfont, FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
QA = ROOT / 'support/workflow_revision3'
OUT.mkdir(exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
W, H = 1200, 835
for bold in [False, True]:
    name = 'DejaVu Sans' + ('-Bold' if bold else '')
    path = findfont(FontProperties(family='DejaVu Sans', weight='bold' if bold else 'normal'))
    pdfmetrics.registerFont(TTFont(name, path))

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>']
tikz = [r'\begin{tikzpicture}[x=1pt,y=-1pt,inner sep=0pt,outer sep=0pt]',
        rf'\path[use as bounding box] (0,0) rectangle ({W},{H});']
texts, boxes, arrows = [], [], []


def label(x, y, plain, tex=None, size=21, bold=False, anchor='middle'):
    tex = plain.replace('&', r'\&') if tex is None else tex
    font = 'DejaVu Sans' + ('-Bold' if bold else '')
    width = pdfmetrics.stringWidth(plain, font, size)
    asc, desc = pdfmetrics.getAscentDescent(font, size)
    left = x-width/2 if anchor == 'middle' else x
    texts.append({'text': plain, 'bbox': [left, y-asc, left+width, y-desc], 'size': size})
    svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="DejaVu Sans" '
               f'font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="black">{escape(plain)}</text>')
    a = 'base' if anchor == 'middle' else 'base west'
    style = r'\sffamily\fontsize{' + str(size) + '}{' + str(size*1.15) + r'}\selectfont'
    if bold:
        style += r'\bfseries'
    tikz.append(rf'\node[anchor={a},text=black,font={{{style}}}] at ({x},{y}) {{{tex}}};')


def rect(x, y, w, h, id=None, lw=1.7):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="black" stroke-width="{lw}"/>')
    tikz.append(rf'\path[fill=white,draw=black,line width={lw}pt] ({x},{y}) rectangle ({x+w},{y+h});')
    if id:
        boxes.append(dict(id=id, x=x, y=y, w=w, h=h))


def node(id, x, y, w, h, lines):
    rect(x, y, w, h, id)
    start = y+h/2-(len(lines)-1)*14+7
    for k, line in enumerate(lines):
        plain, tex = line if isinstance(line, tuple) else (line, None)
        label(x+w/2, start+28*k, plain, tex, bold=(k == 0))


def segment(x1, y1, x2, y2, dashed=False, arrow=True):
    # Single segments only: no hidden short jogs or folded arrow shafts.
    assert x1 == x2 or y1 == y2
    dashsvg = ' stroke-dasharray="7 5"' if dashed else ''
    dashtex = ',dash pattern=on 7pt off 5pt' if dashed else ''
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="1.8"{dashsvg}/>')
    tikz.append(rf'\path[draw=black,line width=1.8pt{dashtex}] ({x1},{y1}) -- ({x2},{y2});')
    if arrow:
        angle = math.atan2(y2-y1, x2-x1)
        ux, uy = math.cos(angle), math.sin(angle)
        pts = [(x2, y2), (x2-9*ux+4.5*uy, y2-9*uy-4.5*ux),
               (x2-9*ux-4.5*uy, y2-9*uy+4.5*ux)]
        svg.append('<polygon points="'+' '.join(f'{x},{y}' for x, y in pts)+'" fill="black"/>')
        tikz.append(r'\path[fill=black,draw=none] '+' -- '.join(f'({x},{y})' for x, y in pts)+r' -- cycle;')
        arrows.append([x1, y1, x2, y2])


def dot(x, y):
    svg.append(f'<circle cx="{x}" cy="{y}" r="5.5" fill="black"/>')
    tikz.append(rf'\fill[black] ({x},{y}) circle[radius=5.5pt];')


label(25, 28, '(a) Paired adaptation from each starting checkpoint', size=25, bold=True, anchor='start')
node('initial', 25, 68, 210, 176, [('Four-step S₀', r'Four-step $S_0$'), 'Parameters', 'and buffers'])
node('adapt2', 305, 68, 250, 64, [('Adapt at T = 2', r'Adapt at $T=2$'), 'K gradient updates'])
node('adapt4', 305, 180, 250, 64, [('Adapt at T = 4', r'Adapt at $T=4$'), 'K gradient updates'])
node('state2', 625, 68, 250, 64, [('S₂(K, s)', r'$S_2(K,s)$'), 'Paired data stream'])
node('state4', 625, 180, 250, 64, [('S₄(K, s)', r'$S_4(K,s)$'), 'Paired data stream'])
node('candidates', 945, 68, 230, 176, ['Candidates', ('S₀, S₂, S₄', r'$S_0,\ S_2,\ S_4$'), 'At each budget'])
for y in [100, 212]:
    segment(235, y, 305, y)
    segment(555, y, 625, y)
    segment(875, y, 945, y)
label(590, 158, 'Retain S₀', r'Retain $S_0$', size=20)
segment(235, 170, 945, 170, dashed=True)
label(25, 280, 'Language: SmoothSpike / WikiText-2', bold=True, anchor='start')
label(615, 280, '3 paired seeds; K = 512, 4,096', r'3 paired seeds; $K=512,\ 4{,}096$', anchor='start')
label(25, 316, 'Vision: SpikingResformer / CIFAR-100', bold=True, anchor='start')
label(615, 316, 'A: 5 paired seeds; B: 3 paired seeds', anchor='start')
label(615, 346, 'K = 512, 1,024, 2,048', r'$K=512,\ 1{,}024,\ 2{,}048$', anchor='start')

label(25, 397, '(b) Raw checkpoint cross-evaluation', size=24, bold=True, anchor='start')
label(645, 397, '(c) BN calibration × inference (vision)',
      r'(c) BN calibration $\times$ inference (vision)', size=24, bold=True, anchor='start')

# Raw source-by-inference matrix, common to language and vision.
label(290, 432, 'One checkpoint per row; two inference windows', size=20)
label(266, 472, 'Inference e = 2', r'Inference $e=2$', size=21)
label(469, 472, 'e = 4', r'$e=4$', size=21)
for y, plain, tex, q2, q4 in [
    (514, 'S₀', r'$S_0$', ('Q₀(2)', r'$Q_0(2)$'), ('Q₀(4)', r'$Q_0(4)$')),
    (582, 'S₂', r'$S_2$', ('Q₂,₂', r'$Q_{2,2}$'), ('Q₂,₄', r'$Q_{2,4}$')),
    (650, 'S₄', r'$S_4$', ('Q₄,₂', r'$Q_{4,2}$'), ('Q₄,₄', r'$Q_{4,4}$'))]:
    label(70, y+7, plain, tex, size=23, bold=True)
    node('raw_'+plain+'_2', 173, y-25, 185, 50, [q2])
    node('raw_'+plain+'_4', 376, y-25, 185, 50, [q4])
label(290, 709, 'P₂ = Q₄,₂ − Q₂,₂', r'$P_2=Q_{4,2}-Q_{2,2}$')
label(290, 739, 'P₄ = Q₂,₄ − Q₄,₄', r'$P_4=Q_{2,4}-Q_{4,4}$')
label(290, 769, 'Positive P: target adaptation has lower CE', r'Positive $P$: target adaptation has lower CE', size=19)

# The BN design is a matrix of actual interventions, not a generic prose card.
label(760, 437, 'K = 512, 1,024', r'$K=512,\ 1{,}024$', bold=True)
label(760, 465, 'Primary subset C1', size=20)
label(1070, 437, 'S₀ and K = 2,048', r'$S_0$ and $K=2{,}048$', bold=True)
label(1070, 465, 'Subsets C1, C2, C3', size=20)
for base, full in [(682, False), (992, True)]:
    label(base+40, 511, 'e = 2', r'$e=2$', size=20)
    label(base+124, 511, 'e = 4', r'$e=4$', size=20)
    label(base-41, 562, 'c = 2', r'$c=2$', size=20)
    label(base-41, 622, 'c = 4', r'$c=4$', size=20)
    rect(base, 530, 168, 120, lw=1.4)
    segment(base+84, 530, base+84, 650, arrow=False)
    segment(base, 590, base+168, 590, arrow=False)
    dot(base+42, 560)
    dot(base+126, 620)
    if full:
        dot(base+126, 560)
        dot(base+42, 620)
    else:
        label(base+126, 567, '—', r'---', size=20)
        label(base+42, 627, '—', r'---', size=20)
label(760, 690, 'Matched windows', size=20)
label(1070, 690, 'Full 2 × 2 × 3 subsets', r'Full $2\times2\times3$ subsets', size=20)
label(912, 727, 'Each source: S₀, S₂, S₄; parameters held fixed',
      r'Each source: $S_0, S_2, S_4$; parameters held fixed', size=20)
label(912, 758, 'Each subset: 2,048 training images', size=20)
segment(25, 784, 1175, 784, arrow=False)
label(600, 818, 'Fixed test inputs • batch = 4 • same samples for every state',
      r'Fixed test inputs $\bullet$ batch $=4$ $\bullet$ same samples for every state', size=21)

svg.append('</svg>')
tikz.append(r'\end{tikzpicture}')
svg_text = '\n'.join(svg)
tikz_text = '\n'.join(tikz)
(OUT/'workflow_revision3.svg').write_text(svg_text, encoding='utf-8')
(OUT/'workflow_revision3.tikz').write_text(tikz_text+'\n', encoding='utf-8')
embedded = ('% BEGIN EMBEDDED FIGURE workflow_revision3\n'
            '\\begingroup\\makeatletter\n\\resizebox{\\linewidth}{!}{%\n'+tikz_text+
            '\n}\n\\endgroup\n% END EMBEDDED FIGURE workflow_revision3\n')
(OUT/'workflow_revision3.embedded.tex').write_text(embedded, encoding='utf-8')
(QA/'standalone.tex').write_text(
    r'\documentclass[border=0pt]{standalone}'+'\n'+
    r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{tikz}\usepackage{graphicx}'+'\n'+
    r'\begin{document}\resizebox{510pt}{!}{\input{../../figures/workflow_revision3.tikz}}\end{document}',
    encoding='utf-8')
run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'standalone.tex'],
                     cwd=QA, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(QA/'compile_output.txt').write_text(run.stdout, encoding='utf-8')
assert run.returncode == 0, run.stdout[-3000:]
pdf = (QA/'standalone.pdf').read_bytes()
(OUT/'workflow_revision3.pdf').write_bytes(pdf)
doc = fitz.open(stream=pdf, filetype='pdf')
doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(OUT/'workflow_revision3.png')

def intersects(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])

text_overlap = []
for i, a in enumerate(texts):
    for b in texts[i+1:]:
        if intersects(a['bbox'], b['bbox']):
            text_overlap.append([a['text'], b['text']])
edge_overlap = []
for e in arrows:
    box = [min(e[0], e[2])-1, min(e[1], e[3])-1, max(e[0], e[2])+1, max(e[1], e[3])+1]
    for text in texts:
        if intersects(box, text['bbox']):
            edge_overlap.append(text['text'])
out_of_bounds = [t['text'] for t in texts if t['bbox'][0] < 0 or t['bbox'][1] < 0 or t['bbox'][2] > W or t['bbox'][3] > H]
audit = dict(status='geometry_passed', format='native editable SVG and TikZ',
             text_overlaps=text_overlap, arrow_text_overlaps=edge_overlap,
             out_of_bounds=out_of_bounds, arrows=len(arrows), all_arrows_single_straight_segments=True,
             colors=['black','white'], timing_branch_removed=True,
             minimum_font_pt_at_510pt_width=min(t['size'] for t in texts)*510/W,
             pdf_images=len(doc[0].get_images()), svg_raster_images=svg_text.count('<image'),
             width_pt=doc[0].rect.width, height_pt=doc[0].rect.height,
             protocol=dict(language=dict(paired_seeds=3, budgets=[512,4096]),
                           vision=dict(paired_seeds=dict(A=5,B=3),budgets=[512,1024,2048]),
                           intermediate_BN='matched-window cells; primary subset only',
                           initial_and_final_BN='full 2 calibration x 2 inference x 3 subsets',
                           calibration_images_per_subset=2048))
(QA/'geometry.json').write_text(json.dumps(audit,indent=2), encoding='utf-8')
assert not(text_overlap or edge_overlap or out_of_bounds), audit
print(json.dumps(audit, indent=2))
