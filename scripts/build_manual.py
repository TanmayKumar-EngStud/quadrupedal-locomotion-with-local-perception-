"""Rebuild the PDF from docs/Manual.md (requires reportlab)."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted,PageBreak,KeepTogether
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/Manual.pdf';OUT.parent.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ManualTitle',fontName='Helvetica-Bold',fontSize=25,leading=30,textColor=colors.HexColor('#173a34'),spaceAfter=16))
styles.add(ParagraphStyle(name='ManualSection',fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=colors.HexColor('#173a34'),spaceBefore=10,spaceAfter=8))
styles.add(ParagraphStyle(name='ManualBody',fontName='Helvetica',fontSize=9.6,leading=14,spaceAfter=9,textColor=colors.HexColor('#263a39')))
styles.add(ParagraphStyle(name='ManualCell',fontName='Helvetica',fontSize=8.2,leading=11,spaceAfter=0))
styles.add(ParagraphStyle(name='ManualReference',fontName='Helvetica',fontSize=7.8,leading=10,spaceAfter=4))
styles.add(ParagraphStyle(name='ManualCode',fontName='Courier',fontSize=7.6,leading=10.5,backColor=colors.HexColor('#edf2ef'),borderPadding=8,spaceBefore=6,spaceAfter=12))

def fmt(text):
    text=escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
    return text

story=[]
for section in (ROOT/'docs/Manual.md').read_text().split('---PAGE---'):
    if story:story.append(PageBreak())
    lines=section.strip().splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            story.append(Preformatted('\n'.join(code),styles['ManualCode']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[- :]+',x) for x in cells):rows.append([Paragraph(fmt(c),styles['ManualCell']) for c in cells])
                i+=1
            width=A4[0]-96;n=len(rows[0]);widths=[width*.25,width*.25,width*.5] if n==3 else [width/n]*n
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9e1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f4f7f3'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8fa99b')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7)]))
            story.extend([table,Spacer(1,12)]);continue
        if line.startswith('# '):story.append(Paragraph(fmt(line[2:]),styles['ManualTitle']));i+=1;continue
        if line.startswith('##'):story.append(Paragraph(fmt(line.lstrip('# ')),styles['ManualSection']));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','- ')) and not re.match(r'^\d+\.',lines[i]):
            para.append(lines[i]);i+=1
        text=' '.join(para)
        if text.startswith('- '):text='&#8226; '+fmt(text[2:])
        else:text=fmt(text)
        story.append(Paragraph(text,styles['ManualReference'] if re.match(r'^\[\d\]',text) or text.startswith('Project sources:') else styles['ManualBody']))


def decorate(canvas,doc):
    canvas.saveState();w,h=A4
    canvas.setFillColor(colors.HexColor('#527568'));canvas.setFont('Helvetica',8)
    canvas.drawString(48,h-28,'GO1 NAVIGATION / PROJECT MANUAL')
    canvas.setStrokeColor(colors.HexColor('#cedbd2'));canvas.line(48,h-37,w-48,h-37)
    canvas.drawString(48,24,'Implementation guide | 14 October 2026')
    canvas.drawRightString(w-48,24,str(doc.page));canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=55,bottomMargin=45,title='Manual - Quadrupedal Navigation',author='Tanmay Kumar, Amedeo Zaccone and Damien Bardina')
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
print(OUT)
