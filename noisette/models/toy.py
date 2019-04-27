from math import sin, cos

def linear_f(xt,A):
    fx = A*xt
    return fx

def sinus_f(xt,a):
    fx = sin(a*xt)
    return fx

def kitagawa_f(xt,t,a,b,c):
    fx = a*xt + b*xt/(1+xt**xt) + c*cos(1.2*t)
    
def linear_h(xt,H):
    hx = H*xt
    return hx

def sinus_h(xt,H):
    hx = H*xt
    return hx

def kitagawa_h(xt,d):
    hx = d*xt**xt
    return hx