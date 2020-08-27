from gl import *

def flat(render, **kwargs):
    A, B, C = kwargs['verts']
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx,ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    
    normal = np.cross(np.subtract(B, A), np.subtract(C, A))
    normal = normal / np.linalg.norm(normal)
    intensity = np.dot(normal, render.light)

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0 :
        return r, g, b
    else:
        return 0,0,0

def unlit(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx,ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    return r, g, b

def gourad(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    intensityA = np.dot(na, render.light)
    intensityB = np.dot(nb, render.light)
    intensityC = np.dot(nc, render.light)

    colorA = (r * intensityA, g * intensityA, b * intensityA)
    colorB = (r * intensityB, g * intensityB, b * intensityB)
    colorC = (r * intensityC, g * intensityC, b * intensityC)

    b = colorA[2] * u + colorB[2] * v + colorC[2] * w
    g = colorA[1] * u + colorB[1] * v + colorC[1] * w
    r = colorA[0] * u + colorB[0] * v + colorC[0] * w

    r = 0 if r < 0 else r
    g = 0 if g < 0 else g
    b = 0 if b < 0 else b

    return r, g, b

def phong(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = V3(nx, ny, nz)

    intensity = np.dot(normal, render.light)

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0

def sombreadoCool(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = V3(nx, ny, nz)

    intensity = np.dot(normal, render.light)
    if intensity < 0:
        intensity = 0

    b *= intensity
    g *= intensity
    r *= intensity

    if render.active_texture2:
        texColor = render.active_texture2.getColor(tx, ty)

        b += (texColor[0] / 255) * (1 - intensity)
        g += (texColor[1] / 255) * (1 - intensity)
        r += (texColor[2] / 255) * (1 - intensity)


    return r, g, b

def toon(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = V3(nx, ny, nz)

    intensity = np.dot(normal, render.light)

#variamos intensidad para el toon shader
    if (intensity>0 and intensity<=0.3):
        intensity=0.3
    # if (intensity>0.3 and intensity<=0.6):
    #     intensity=0.6
    # if (intensity>0.6 and intensity<=0.9):
    #     intensity=0.9
    if (intensity>0.3):
        intensity=1

    b *= intensity
    g *= intensity
    r *= intensity


    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0


def infraRojo(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
        if (b>g and b>r):
            r=0.1
            g=0.1
            b=0.7
        if (g>b and g>r):
            r=0.1
            g=0.7
            b=0.1
        if (r>g and r>b):
            r=0.7
            g=0.1
            b=0.1

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = V3(nx, ny, nz)

    intensity = np.dot(normal, render.light)

#variamos intensidad para el toon shader
    if (intensity>0 and intensity<=0.3):
        intensity=0.3
        r=0.1
        g=0.1
        b=0.9
    if (intensity>0.3 and intensity<=0.6):
        intensity=0.6
        r=0.1
        g=0.9
        b=0.5
    if (intensity>0.6 and intensity<=0.9):
        intensity=0.9
    if (intensity>0.9):
        intensity=1
        r=1
        g=1
        b=0

    b *= intensity
    g *= intensity
    r *= intensity


    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0

def freedom(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
        if (b>g and b>r):
            r=0.1
            g=0.1
            b=0.9
        if (g>b and g>r):
            r=0.1
            g=0.1
            b=0.5
        if (r>g and r>b):
            r=0.7
            g=0.1
            b=0.1

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = V3(nx, ny, nz)

    intensity = np.dot(normal, render.light)

#variamos intensidad para el toon shader
    if (intensity>0 and intensity<=0.3):
        intensity=0.3
        r=0.1
        g=0.1
        b=0.9
    if (intensity>0.3 and intensity<=0.6):
        intensity=0.6
        r=0.1
        g=0.1
        b=0.7
    if (intensity>0.6 and intensity<=0.9):
        intensity=0.9
    if (intensity>0.9):
        intensity=1
        r=1
        g=1
        b=1

    b *= intensity
    g *= intensity
    r *= intensity


    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0

def normalMap(render, **kwargs):
    A, B, C = kwargs['verts']
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    tx = ta.x * u + tb.x * v + tc.x * w
    ty = ta.y * u + tb.y * v + tc.y * w

    if render.active_texture:
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w
    normal = V3(nx, ny, nz)

    if render.active_normalMap:
        texNormal = render.active_normalMap.getColor(tx, ty)
        texNormal = [ (texNormal[2] / 255) * 2 - 1,
                      (texNormal[1] / 255) * 2 - 1,
                      (texNormal[0] / 255) * 2 - 1]

        texNormal = texNormal / np.linalg.norm(texNormal)

        edge1 = np.subtract(B,A)
        edge2 = np.subtract(C,A)
        deltaUV1 = np.subtract(tb, ta)
        deltaUV2 = np.subtract(tc, ta)
        tangent = [0,0,0]
        f = 1 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])
        tangent[0] = f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0])
        tangent[1] = f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1])
        tangent[2] = f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
        tangent = tangent / np.linalg.norm(tangent)
        tangent = np.subtract(tangent, np.multiply(np.dot(tangent, normal),normal))
        tangent = tangent / np.linalg.norm(tangent)

        bitangent = np.cross(normal, tangent)
        bitangent = bitangent / np.linalg.norm(bitangent)

        #para convertir de espacio global a espacio tangente
        tangentMatrix = matrix([[tangent[0],bitangent[0],normal[0]],
                                [tangent[1],bitangent[1],normal[1]],
                                [tangent[2],bitangent[2],normal[2]]])

        light = render.light
        light = tangentMatrix @ light
        light = light.tolist()[0]
        light = light / np.linalg.norm(light)

        intensity = np.dot(texNormal, light)
    else:
        intensity = np.dot(normal, render.light)

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0


