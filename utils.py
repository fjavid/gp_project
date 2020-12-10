import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1).repeat(1, C, H, W).to(device)
    interpolated_images = epsilon * real + (1. - epsilon) * fake
    # Mix critic scores
    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1.)*(gradient_norm - 1.))
    return gradient_penalty

class Mesh:
    def __init__(self, V, F):
        self.V = V
        self.F = F
    def setFaceNormals():
        self.N = []
        for face in F:
            e0 = numpy.array(V[face[1]]-V[face[0]])
            e2 = numpy.array(V[face[2]]-V[face[0]])
            n = numpy.cross(e0, e3)
            N.append(list(n))

def loadOBJ(fullpath):
    V = []
    F = []
    with open(fullpath, 'r') as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
        data = line.split(' ')
        if data[0] == 'v':
            V.append([float(data[1]), float(data[2]), float(data[3])])
        elif data[0] == 'f':
            F.append([int(data[1])-1, int(data[2])-1, int(data[3])-1])
    
    return Mesh(V, F)