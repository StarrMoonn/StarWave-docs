"""Regenerate original vector schematics; no simulation data or benchmark bars.

Requires fonttools and the Noto Sans CJK SC regular font (OFL 1.1).
Labels are outlined for consistent Chinese/English rendering on any browser.
"""
import argparse
from html import escape
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / '_static' / 'reconstruction'
BLUE, TEAL, ORANGE, INK, MUTED = '#256b99', '#187f7a', '#b45d1b', '#243d50', '#526979'

class Canvas:
    def __init__(self, name, locale, height, title, desc, font):
        self.name, self.locale, self.height, self.font = name, locale, height, font
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" viewBox="0 0 760 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#526979"/></marker></defs>',
            '<rect width="760" height="100%" fill="#ffffff"/>']
        self.text(28, 36, title, 22, INK)
        self.line(28, 53, 732, 53, '#dbe5ea')
    def rect(self,x,y,w,h,fill='#f5f8fa',stroke='#d0dce4',r=8):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
    def line(self,x1,y1,x2,y2,color=MUTED,width=1.5,arrow=False,dash=False):
        self.parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="5 5"' if dash else '')+'/>')
    def poly(self,pts,fill,stroke=BLUE):
        self.parts.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    def text(self,x,y,label,size=16,color=INK,anchor='start'):
        fonts=[next(f for f in self.font if ord(c) in f.getBestCmap()) for c in label]
        scales=[size/f['head'].unitsPerEm for f in fonts]
        names=[f.getBestCmap()[ord(c)] for c,f in zip(label,fonts)]
        advances=[f['hmtx'][name][0]*scale for f,name,scale in zip(fonts,names,scales)]
        width=sum(advances)
        x-=width/2 if anchor=='middle' else width if anchor=='end' else 0
        self.parts.append(f'<g aria-label="{escape(label,quote=True)}" data-font-size="{size}" fill="{color}">')
        for f,name,scale,advance in zip(fonts,names,scales,advances):
            glyphs=f.getGlyphSet(); pen=SVGPathPen(glyphs); glyphs[name].draw(pen)
            if pen.getCommands(): self.parts.append(f'<path d="{pen.getCommands()}" transform="translate({x:.3f} {y}) scale({scale:.6f} {-scale:.6f})"/>')
            x+=advance
        self.parts.append('</g>')
    def lines(self,x,y,labels,size=16,color=INK,step=25,anchor='start'):
        for j,label in enumerate(labels): self.text(x,y+j*step,label,size,color,anchor)
    def box(self,x,y,w,h,labels,fill='#edf5fa',stroke=BLUE):
        self.rect(x,y,w,h,fill,stroke)
        self.lines(x+w/2,y+h/2-(len(labels)-1)*12+6,labels,16,INK,24,'middle')
    def finish(self):
        self.line(28,self.height-37,732,self.height-37,'#dbe5ea')
        self.text(28,self.height-14,'StarWave · '+('实现示意，非模拟结果' if self.locale=='zh' else 'Implementation schematic, not simulation data'),16,MUTED)
        self.parts.append('</svg>'); OUT.mkdir(parents=True,exist_ok=True)
        (OUT/f'{self.name}-{self.locale}.svg').write_text('\n'.join(self.parts)+'\n')

def diagrams(font, lang):
    zh=lang=='zh'
    t=lambda en,cn: cn if zh else en
    c=Canvas('flow',lang,540,t('One forward run, two backward trajectories','一次正演，两条反向轨迹'),t('Forward propagation records a boundary tape and final physical state. During backward, the reconstructed forward field and the separate adjoint field meet in material-gradient accumulation.','正演记录边界带与终态物理场。反向计算时，重建的正演场与独立传播的伴随场在材料梯度累加处耦合。'),font)
    c.text(28,84,t('FORWARD  ·  n = 0 → N − 1','正向  ·  n = 0 → N − 1'),17,BLUE)
    c.box(28,107,164,72,[t('Model + source','模型 + 震源')]);c.box(247,107,220,72,[t('Staggered CUDA update','交错网格 CUDA 更新')]);c.box(522,107,210,72,[t('Predicted records','预测记录'),t('Terminal physical state','终态物理场')])
    c.line(195,143,242,143,arrow=True);c.line(470,143,517,143,arrow=True)
    c.box(247,217,220,63,[t('Pre-step boundary tape','每步更新前的边界带')],'#fff4e7',ORANGE);c.line(357,183,357,213,arrow=True)
    c.text(28,318,t('RECONSTRUCT','重建'),17,TEAL)
    c.box(28,343,223,71,[t('Reconstructed forward field','重建的正演场'),t('Restore + undo updates','恢复边界 + 撤销更新')],'#eaf7f4',TEAL)
    c.box(293,343,185,71,[t('Material gradients','材料参数梯度'),t('λ, μ, buoyancy','λ、μ、浮力系数')],'#f6f3fa','#80699c')
    c.box(521,343,211,71,[t('Adjoint field','伴随场'),t('Transpose updates','转置更新')],'#edf5fa',BLUE)
    c.line(250,378,287,378,arrow=True);c.line(518,378,484,378,arrow=True)
    c.line(310,281,216,338,arrow=True)
    c.line(555,182,555,291,dash=True);c.line(555,291,244,339,arrow=True,dash=True)
    c.text(485,262,t('terminal state','终态物理场'),16,MUTED)
    c.line(676,182,676,338,arrow=True,dash=True)
    c.text(547,323,t('loss cotangent','损失余切'),16,MUTED)
    c.lines(28,456,[t('Same time index, different roles. Reconstruction supplies saved forward information.','同一时间层，不同物理角色。重建负责提供正演信息。'),t('The adjoint carries loss sensitivity; their contractions produce model gradients.','伴随场携带目标函数敏感度；二者的收缩产生模型梯度。')],16)
    c.finish()

    c=Canvas('domain',lang,624,t('What the boundary tape actually stores','边界带实际保存什么'),t('Two- and three-dimensional domains show a PML velocity shell and narrow direction-specific traction strips. Velocity-shell intersections are stored once, while traction blocks may overlap.','二维与三维区域中的 PML 速度壳层和按方向保存的窄牵引带。速度壳层交叠处去重，牵引分块允许交叠。'),font)
    c.text(28,88,t('2D  ·  [y, x]','二维  ·  [y, x]'),18,BLUE);c.text(403,88,t('3D  ·  [z, y, x]','三维  ·  [z, y, x]'),18,BLUE)
    c.rect(46,110,290,223,'#dcecf7',BLUE,0);c.rect(94,153,194,137,'#ffffff','#718895',0)
    for x,y,w,h in [(46,141,290,12),(46,290,290,12),(82,110,12,223),(288,110,12,223)]:c.rect(x,y,w,h,'#f8d5a9',ORANGE,0)
    c.lines(191,204,[t('Undamped','无阻尼'),t('interior','内部区域')],17,INK,25,'middle')
    c.line(57,353,113,353,arrow=True);c.text(122,359,'x',17);c.line(29,115,29,168,arrow=True);c.text(21,187,'y',17)
    c.poly('425,164 502,115 708,115 631,164','#e9f2f8');c.poly('631,164 708,115 708,296 631,345','#c8deee');c.poly('425,164 631,164 631,345 425,345','#dfedf7')
    c.poly('454,187 607,187 607,318 454,318','#ffffff','#718895');c.poly('425,180 631,180 631,187 425,187','#f8d5a9',ORANGE);c.poly('425,318 631,318 631,325 425,325','#f8d5a9',ORANGE)
    c.poly('447,164 454,164 454,345 447,345','#f8d5a9',ORANGE);c.poly('607,164 614,164 614,345 607,345','#f8d5a9',ORANGE)
    c.lines(531,239,[t('Physical','物理'),t('interior','内部区域')],17,INK,25,'middle')
    c.line(660,338,708,308,arrow=True);c.text(714,306,'y',17);c.line(639,360,698,360,arrow=True);c.text(710,366,'x',17);c.line(410,280,410,332,arrow=True);c.text(396,350,'z',17)
    c.rect(28,387,17,17,'#dcecf7',BLUE,0);c.text(55,402,t('Velocity shell: all d velocity components','速度壳层：全部 d 个速度分量'),16)
    c.rect(28,421,17,17,'#f8d5a9',ORANGE,0);c.text(55,436,t('Traction strips: only stresses needed by each face','牵引窄带：仅保存对应面需要的应力分量'),16)
    c.lines(28,479,[t('2D: y-face (σyy, σxy); x-face (σxy, σxx)','二维：y 面 (σyy, σxy)；x 面 (σxy, σxx)'),t('3D: z-face (σzz, σyz, σxz); y-face (σyz, σyy, σxy)','三维：z 面 (σzz, σyz, σxz)；y 面 (σyz, σyy, σxy)'),t('         x-face (σxz, σxy, σxx)','         x 面 (σxz, σxy, σxx)')],16,INK,27)
    c.text(28,568,t('Strip depth ≤ stencil radius. High-side half-grid plane is included.','窄带厚度不超过差分半径；布局包含高侧半网格平面。'),16,MUTED)
    c.finish()

    c=Canvas('timeline',lang,575,t('Reverse the update order; align the gradient','倒序撤销更新，对齐梯度时间层'),t('A pre-step state contains staggered velocity and stress. Forward first updates velocity then stress. Reverse reconstruction subtracts the stress source, reverses stress, subtracts velocity sources and reverses velocity, with tape restoration. The adjoint applies transposed operators.','步前状态包含交错时间层的速度和应力。正演先更新速度再更新应力；重建依次撤销应力源、应力更新、速度源与速度更新，并恢复边界带。伴随计算应用转置算子。'),font)
    c.text(28,88,t('FORWARD','正演'),17,BLUE)
    c.box(28,110,180,78,['xⁿ = (vⁿ⁻½, σⁿ)',t('save tape Bⁿ','保存边界带 Bⁿ')])
    c.box(259,110,218,78,[t('Velocity + force source','速度更新 + 力源'),'vⁿ⁺½'])
    c.box(528,110,204,78,[t('Stress + pressure source','应力更新 + 压力源'),'σⁿ⁺¹'])
    c.line(210,149,253,149,arrow=True);c.line(480,149,522,149,arrow=True)
    c.text(28,232,t('RECONSTRUCT THE FORWARD STATE','重建正演状态'),17,TEAL)
    c.box(28,255,213,95,[t('Undo velocity source','撤销速度源'),t('Reverse velocity update','逆向速度更新'),t('Restore shell vⁿ⁻½','恢复壳层 vⁿ⁻½')],'#eaf7f4',TEAL)
    c.box(277,255,214,95,[t('Undo pressure source','撤销压力源'),t('Reverse stress update','逆向应力更新'),t('Read traction strips Bⁿ','读取牵引带 Bⁿ')],'#eaf7f4',TEAL)
    c.box(527,255,205,95,['xⁿ⁺¹ = (vⁿ⁺½, σⁿ⁺¹)',t('terminal at n = N − 1','n = N − 1 时从终态起步')],'#eaf7f4',TEAL)
    c.line(524,303,498,303,arrow=True);c.line(273,303,246,303,arrow=True)
    c.line(629,192,629,249,arrow=True)
    c.rect(28,386,704,110,'#f6f3fa','#c5b5d7')
    c.text(47,415,t('ADJOINT ≠ RECONSTRUCTION','伴随传播 ≠ 正演场重建'),17,'#60467f')
    c.lines(47,447,[t('Adjoint: transpose each discrete update + inject receiver cotangents.','伴随：转置离散更新，并注入接收记录的余切。'),t('Gradient: contract at matching substeps; λ/μ also need the PML filter transpose.','梯度：在相应子步收缩；λ/μ 还需 PML 滤波器的转置。')],16)
    c.finish()

    c=Canvas('memory',lang,568,t('Volume history → boundary tape + working states','全体积历史 → 边界带 + 工作状态'),t('Schematic allocation comparison, not measured bar lengths: full stores requested derivative histories throughout the volume at sampled times; boundary stores a per-step boundary tape, terminal physical fields and bounded-volume workspaces. Both retain other inputs and outputs.','非实测柱状图：full 按采样时间保存所需的全体积导数历史；boundary 保存逐步边界带、终态物理场及体积级工作区。两种模式都仍需输入与输出。'),font)
    c.text(28,90,t('FULL HISTORY','全历史 full'),17,BLUE);c.text(409,90,t('BOUNDARY RECONSTRUCTION','边界重建 boundary'),17,TEAL)
    for k in range(4):
        x=58+k*28;y=136-k*10;c.rect(x,y,168,131,'#dcecf7',BLUE,3)
        for q in range(1,5):c.line(x+q*28,y,x+q*28,y+131,'#a7c9df',1);c.line(x,y+q*22,x+168,y+q*22,'#a7c9df',1)
    c.text(202,301,t('sampled volume histories','采样后的体积历史'),16,INK,'middle')
    for k in range(4):
        x=421+k*24;y=136-k*10;c.rect(x,y,170,131,'#dcecf7',BLUE,0);c.rect(x+20,y+19,130,93,'#ffffff',BLUE,0)
    c.text(554,301,t('every-step boundary tape','逐内部时间步的边界带'),16,INK,'middle')
    c.line(372,112,372,494,'#dbe5ea',1)
    c.lines(28,345,['O(B × (N/s) × V_pad × H)',t('H: requested image-history channels','H：需要的梯度历史通道'),t('s: material-gradient sampling stride','s：材料梯度采样步长')],17,INK,30)
    c.lines(403,345,['O(B × N × K) + O(B × V)',t('K: velocity shell + traction strips','K：速度壳层 + 牵引带'),t('Plus terminal state and workspace','另有终态和工作区')],17,INK,30)
    c.lines(28,458,[t('Fixed shell thickness: K scales with boundary area, not domain volume.','固定壳层厚度时，K 随边界面积增长，而非随区域体积增长。'),t('Time and shot count still multiply storage. Reconstruction adds computation.','时间长度与炮数仍成倍增加存储；重建需要额外计算。')],16)
    c.finish()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font',type=Path,default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    parser.add_argument('--font-index',type=int,default=2)
    args=parser.parse_args();font=[TTFont(args.font,fontNumber=args.font_index), TTFont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    for locale in ('en','zh'):diagrams(font,locale)
