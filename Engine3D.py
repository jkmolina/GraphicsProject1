from gl import Render, color, V2, V3
from obj import Obj, Texture
from shaders import *

r = Render(1920,1080)

#usamos la textura de dojo para hacer el fondo

texture=Texture('./models/dojo.bmp')

r.glBackground(texture)


#katana (phong)
r.active_shader=phong
posModel = V3( 2, 0, -5)

r.active_texture = Texture('./models/Albedo.bmp')

r.loadModel('./models/katana.obj', posModel, V3(4,4,4), V3(0,0,270))

#cleaver (freedom)

r.active_shader=freedom
posModel = V3( -3, 0, -5)

r.active_texture = Texture('./models/cleaver.bmp')

r.loadModel('./models/cleaver.obj', posModel, V3(10,10,10), V3(90,0,-30))

#dagger (infrarojo)

r.active_shader=infraRojo
posModel = V3( 0, 2, -5)

r.active_texture = Texture('./models/dagger.bmp')

r.loadModel('./models/dagger.obj', posModel, V3(0.07,0.07,0.07), V3(45,0,0))

#sting (normalMap)

r.active_shader=normalMap
posModel=V3(-1.5,-2,-5)

r.active_texture = Texture('./models/stingNormal.bmp')

r.loadModel('./models/sting.obj', posModel, V3(0.06,0.06,0.06), V3(30,90,0))

#crossbow (toon)

r.active_shader=toon
posModel=V3(2.5,-1.7,-5)

r.active_texture = Texture('./models/crossbow.bmp')

r.loadModel('./models/crossbow.obj', posModel, V3(0.04,0.04,0.04), V3(30,90,0))

r.glFinish('output.bmp')
