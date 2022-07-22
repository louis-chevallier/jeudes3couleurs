import os, sys, pickle
import torch
import numpy as _onp
import jax.numpy as _jnp

from jax import jit as jjit
import jax

tnp = _onp

from utillc import EKO, EKON, TYPE

S : int = 1
colors = { 1,2,3}
node = tnp.array

def getboard(x) : return x

gx, gy = tnp.mgrid[0:S+2, 0:S+2]
z = tnp.zeros((S+2,S+2)).astype(tnp.int16)
mask = tnp.logical_or(tnp.logical_or(gx == 0, gx == S+1), tnp.logical_or(gy == 0, gy == S+1))

v4 = tnp.array([[ False, True, False],
               [ True, False, True],
               [ False, True, False]])

x: list[int] = [1]

def blank(cc : tnp.int16 = 0) -> node:
    y = tnp.where(mask, z, cc)
    return y

c1 = blank(1)
c2 = blank(2)
c3 = blank(3)

class N :
    def __init__(self, g, father = None, i=0, j=0,c=0) :
        self.g = g
        self.i = i
        self.j = j
        self.c = c
        self.father = father
        self.children = []
        

def normal(g : node) -> node:
    EKON(g)
    v1 = g[1:S+1, 1:S+1] == c1[1:S+1, 1:S+1]
    v2 = g[1:S+1, 1:S+1] == c2[1:S+1, 1:S+1]
    v3 = g[1:S+1, 1:S+1] == c3[1:S+1, 1:S+1]
    ll = []
    if v1.any() : ll.append(v1.argmax())
    if v2.any() : ll.append(v2.argmax())
    if v3.any() : ll.append(v3.argmax())
    EKON(ll)
    le = len(ll)
    EKON(le)
    EKON( ll[-1:])
    EKON( ll[-2:])
    ll = ll + ll[-2:] * (3-le) 
    EKON(ll)        
    l = tnp.array([ll])
    EKON(l)
    EKON(tnp.argsort(l))
    las = tnp.argsort(l)
    EKON(las)
    x = blank()
    for ie, e in enumerate(las) :
        EKON(e)
        ee = e+1
        m = g == ee
        EKON(m)
        #x[m] = ie+1
        x = tnp.where(m, ie+1, x)
        EKON(x)
    EKON(x)
    return x


def possible(area : node) -> set:
    a = getboard(area).copy()
    aa = tnp.where(v4, a, 0).flatten()
    s = colors - set(list(_onp.array(aa)))
    return s

def play(c : tnp.int16, i : int, j : int, x : N) -> N:
    g = x.g
    cc = tnp.logical_and(gx == i, gy == j)
    g1 = tnp.where(cc, c, g)    
    return N(g1, x, i, j, c)

def minbranch(x) :
    if len(x.children) == 0 : return 1
    return min([ minbranch(e) for e in x.children]) + 1

def lose(x : N) -> bool :
    if len(x.children) == 0 : return True
    return all([ not lose(e) for e in x.children])

def next(x : N) -> list[N]:
    game = x.g
    n = []
    for i_ in range(0, S):
        for j_ in range(0, S):
            i,j = i_ + 1, j_+1
            #EKON(game)
            if game[i,j] == 0 :
                l = possible(game[i-1:i+2, j-1:j+2])
                n = n + [ play(m, i, j, x) for m in l]
    return n

def random() -> node :
    g0 = blank()
    g0[1:S+1, 1:S+1] = tnp.random.randint(0, 4, size=(S,S))
    return g0

def valid(game : node) -> tuple[int, int]:
    game = getboard(node)
    for i_ in range(0, S):
        for j_ in range(0, S):
            i,j = i_ + 1, j_+1
            g = game[i-1:i+2, j-1:j+2].copy()
            ss = g[1,1]
            g[1,1] = 0
            l = possible(g)
            #EKON(l, ss)
            if len(l) == 0 and ss > 0: return i,j
            if ss > 0 and ss not in l : return i,j
    return 0,0

        
def dump(l : list[node]) -> str :
    sstr  = lambda x : str(x.item())
    sss = [ "\n".join([ "".join(map(sstr, list(row))) for row in e ]) for e in l ]
    return "\n" + "\n\n".join(sss)

def polarite(g : node) -> int :
    p = g == 0
    s = g.sum()
    return s%2
    
        
if __name__ == '__main__':
    EKO()
    EKON(dump([blank(1)]))


    starting = N(blank())
    
    l = [ starting ]
    EKON(starting.g)
    for k in range(15) :
        l = [ n for g in l for n in next(g)]

        vflip = lambda x : tnp.fliplr(x)
        hflip = lambda x : tnp.flip(x)
        rot = lambda x : tnp.rot90(x)
        roti = lambda x : tnp.rot90(x, axes=(1,0))        
        sss = lambda x : dump([x])

        normaln = lambda x : N(normal(x.g), x.father, x.i, x.j, x.c)
        
        l = list(map(normaln, l))
        ls = set([])
        l1 = []
        for ne in l :
            e = ne.g
            se = sss(e)
            if (se not in ls and
                sss(normal(hflip(e))) not in ls and

                sss(normal(vflip(hflip(e)))) not in ls and
                sss(normal(rot(vflip(e)))) not in ls and
                sss(normal(vflip(rot(e)))) not in ls and                                
                
                sss(normal(vflip(e))) not in ls and
                sss(normal(rot(e)))  not in ls and
                sss(normal(roti(e)))  not in ls) :                
                ls.add(se)
                l1.append(ne)
        l = l1
        for x in l :
            x.father.children.append(x)
            
        EKON(k, len(l))
        for x in l :
            EKON(x.g)
            
        if len(l) == 0 :
            EKON(lose(starting))
            EKON(starting.children)
            EKON(minbranch(starting))
            EKON([ lose(e) for e in starting.children])
            break
        #EKON(l)
        with open("graph.pckl", "wb") as fd :  pickle.dump(l, fd)
    
    """
        
    tnp.random.shuffle(l)
    l = l[0:1]
    #EKON(i, jd3c.dump(l[0:1]))
    EKON(i, j.dump(l))
    if len(l) == 0 :            break
    EKON(j.valid(l[0]))
    nn = j.normal(l[0])
    l = [nn]
    """
